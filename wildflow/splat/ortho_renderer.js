async ({ width, height, metersPerPixel, cameraCenter }) => {

    const waitRenderComplete = () => new Promise((resolve) => {
        const eventHandle = window.scene.events.on('postrender', () => {
            eventHandle.off();
            resolve(true);
        });
    });

    const waitSingleFrame = () => new Promise((resolve) => {
        const eventHandle = window.scene.events.on('postrender', () => {
            eventHandle.off();
            resolve(null);
        });
    });

    const waitForCameraStability = async (maxWaitMs = 2000) => {
        let totalWaitTime = 0;
        const tolerance = 0.001;

        while (totalWaitTime < maxWaitMs) {
            const rotation = window.scene.camera.entity.getLocalEulerAngles();
            const isStable = Math.abs(rotation.x) < tolerance &&
                Math.abs(rotation.y) < tolerance &&
                Math.abs(rotation.z) < tolerance;

            if (isStable) {
                return totalWaitTime;
            }

            await new Promise(resolve => setTimeout(resolve, 100));
            totalWaitTime += 100;

            window.scene.forceRender = true;
            await waitRenderComplete();
        }

        return totalWaitTime;
    };

    window.scene.events.fire('view.setBands', 3);
    window.scene.events.fire('camera.setOverlay', false);
    window.scene.events.fire('camera.setMode', 'rings');

    if (window.scene.events.invoke('grid.visible')) {
        window.scene.events.fire('grid.setVisible', false);
    }

    window.scene.events.fire('camera.setBound', false);
    window.scene.events.fire('view.setOutlineSelection', false);

    const ColorClass = window.pc && window.pc.Color ? window.pc.Color : (window.Color || null);
    if (ColorClass) {
        window.scene.events.fire('setBgClr', new ColorClass(0, 0, 0, 1));
    } else {
        window.scene.events.fire('setBgClr', { r: 0, g: 0, b: 0, a: 1 });
    }

    window.scene.camera.startOffscreenMode(width, height);
    window.scene.camera.renderOverlays = false;

    window.scene.events.fire('camera.align', 'pz');
    window.scene.camera.ortho = true;

    const sceneBounds = window.scene.bound;

    const focalCenter = (() => {
        if (cameraCenter && Number.isFinite(cameraCenter.x) && Number.isFinite(cameraCenter.y)) {
            return {
                x: cameraCenter.x,
                y: cameraCenter.y,
                z: cameraCenter.z || sceneBounds.center.z
            };
        } else {
            return sceneBounds.center;
        }
    })();

    const cameraZPosition = 200.0;
    window.scene.camera.entity.setLocalPosition(focalCenter.x, focalCenter.y, cameraZPosition);
    window.scene.camera.entity.setLocalEulerAngles(0, 0, 180);

    const highAltitudeFocalZ = cameraZPosition - 5.0;
    const lookDownTarget = { x: focalCenter.x, y: focalCenter.y, z: highAltitudeFocalZ };
    window.scene.camera.setFocalPoint(lookDownTarget, 0);

    window.scene.camera.ortho = true;

    await waitForCameraStability();

    {
        const cameraComponent = window.scene.camera.entity.camera;
        const sceneRadius = Math.sqrt(
            sceneBounds.halfExtents.x * sceneBounds.halfExtents.x +
            sceneBounds.halfExtents.y * sceneBounds.halfExtents.y +
            sceneBounds.halfExtents.z * sceneBounds.halfExtents.z
        );
        cameraComponent.nearClip = Math.max(1e-6, -sceneRadius * 4);
        cameraComponent.farClip = sceneRadius * 4;
    }

    await waitForCameraStability();

    const scaledDistance = (() => {
        const cameraComponent = window.scene.camera.entity.camera;
        const currentOrthoHeight = Math.max(1e-12, cameraComponent.orthoHeight);
        const desiredOrthoHeight = height * metersPerPixel * 0.5;
        const currentDistance = window.scene.camera.distance;
        const result = currentDistance * (desiredOrthoHeight / currentOrthoHeight);

        return result;
    })();
    window.scene.camera.setDistance(scaledDistance, 0);

    window.scene.forceRender = true;
    await waitRenderComplete();

    const finalHighZ = 200.0;
    const currentPos = window.scene.camera.entity.getLocalPosition();
    window.scene.camera.entity.setLocalPosition(currentPos.x, currentPos.y, finalHighZ);

    window.scene.forceRender = true;
    await waitRenderComplete();

    const isCanvasAllBlack = (pixelData) => {
        let nonBlackPixels = 0;
        for (let i = 0; i < pixelData.data.length; i += 4) {
            if (pixelData.data[i] > 0 || pixelData.data[i + 1] > 0 || pixelData.data[i + 2] > 0) {
                nonBlackPixels++;
            }
        }
        console.log(`nonBlackPixels: ${nonBlackPixels}`);
        return nonBlackPixels === 0;
    };

    const getPixelDataWithRetry = async (maxRetries = 4, retryDelayMs = 2500) => {
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            if (attempt > 1) {
                await waitForCameraStability(2000);
                for (let i = 0; i < 5; i++) {
                    window.scene.forceRender = true;
                    await waitRenderComplete();
                }
            }

            window.scene.forceRender = true;
            await waitRenderComplete();

            const renderTarget = window.scene.camera.entity.camera.renderTarget;
            const colorBuffer = renderTarget.colorBuffer;
            const data = new Uint8Array(colorBuffer.width * colorBuffer.height * 4);
            const pixelData = await colorBuffer.read(0, 0, colorBuffer.width, colorBuffer.height, {
                renderTarget: renderTarget,
                data: data
            }).then(() => ({ data, bufferWidth: colorBuffer.width, bufferHeight: colorBuffer.height }));

            if (!isCanvasAllBlack(pixelData)) return pixelData;

            if (attempt < maxRetries) {
                await new Promise(resolve => setTimeout(resolve, retryDelayMs));
            }
        }

        const renderTarget = window.scene.camera.entity.camera.renderTarget;
        const colorBuffer = renderTarget.colorBuffer;
        const data = new Uint8Array(colorBuffer.width * colorBuffer.height * 4);
        return colorBuffer.read(0, 0, colorBuffer.width, colorBuffer.height, {
            renderTarget: renderTarget,
            data: data
        }).then(() => ({ data, bufferWidth: colorBuffer.width, bufferHeight: colorBuffer.height }));
    };

    const pixelData = await getPixelDataWithRetry();

    const canvas = (() => {
        const canvas = document.createElement('canvas');
        canvas.width = pixelData.bufferWidth;
        canvas.height = pixelData.bufferHeight;
        const canvasContext = canvas.getContext('2d');

        canvasContext.fillStyle = '#000000';
        canvasContext.fillRect(0, 0, pixelData.bufferWidth, pixelData.bufferHeight);

        const processedPixels = (() => {
            const tempRow = new Uint8ClampedArray(pixelData.bufferWidth * 4);
            const flippablePixels = new Uint8ClampedArray(pixelData.data.buffer);

            for (let rowIndex = 0; rowIndex < Math.floor(pixelData.bufferHeight / 2); rowIndex++) {
                const topRowStart = rowIndex * pixelData.bufferWidth * 4;
                const bottomRowStart = (pixelData.bufferHeight - rowIndex - 1) * pixelData.bufferWidth * 4;
                tempRow.set(flippablePixels.subarray(topRowStart, topRowStart + pixelData.bufferWidth * 4));
                flippablePixels.copyWithin(topRowStart, bottomRowStart, bottomRowStart + pixelData.bufferWidth * 4);
                flippablePixels.set(tempRow, bottomRowStart);
            }

            for (let pixelIndex = 0; pixelIndex < flippablePixels.length; pixelIndex += 4) {
                flippablePixels[pixelIndex + 3] = 255;
            }

            return flippablePixels;
        })();

        const finalImageData = new ImageData(processedPixels, pixelData.bufferWidth, pixelData.bufferHeight);
        canvasContext.putImageData(finalImageData, 0, 0);
        return canvas;
    })();

    window.scene.camera.endOffscreenMode();
    window.scene.camera.renderOverlays = true;
    window.scene.forceRender = true;
    await waitSingleFrame();

    return canvas.toDataURL('image/png');
}

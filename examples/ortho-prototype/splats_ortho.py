#!/usr/bin/env python3
"""
Standalone orthographic renderer for PLY files using SuperSplat.
Single file - no boilerplate, just import and call!
"""

import asyncio
import base64
import os
import socket
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests
from playwright.async_api import async_playwright


class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with CORS headers for PLY files."""
    
    def __init__(self, *args, serve_directory: str = None, **kwargs):
        self.serve_directory = serve_directory or os.getcwd()
        super().__init__(*args, directory=self.serve_directory, **kwargs)
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()
    
    def guess_type(self, path):
        if path.endswith('.ply'):
            return 'application/ply'
        return super().guess_type(path)
    
    def log_message(self, format, *args):
        pass  # Silent


class PLYServer:
    """Simple HTTP server for PLY files."""
    
    def __init__(self, serve_directory: str, port: int = 8123):
        self.serve_directory = Path(serve_directory).resolve()
        self.port = port
        self.server = None
        self.server_thread = None
        self._running = False
    
    def _find_free_port(self, start_port: int) -> int:
        port = start_port
        while port < start_port + 100:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                port += 1
        raise RuntimeError(f"Could not find free port starting from {start_port}")
    
    def start(self) -> str:
        if self._running:
            return f"http://127.0.0.1:{self.port}"
        
        self.port = self._find_free_port(self.port)
        
        def handler_factory(*args, **kwargs):
            return CORSHTTPRequestHandler(*args, serve_directory=str(self.serve_directory), **kwargs)
        
        self.server = HTTPServer(('127.0.0.1', self.port), handler_factory)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self._running = True
        time.sleep(0.1)
        
        return f"http://127.0.0.1:{self.port}"
    
    def stop(self):
        if self.server and self._running:
            self.server.shutdown()
            self.server.server_close()
            if self.server_thread:
                self.server_thread.join(timeout=5.0)
            self._running = False
    
    def get_file_url(self, filename: str) -> str:
        return f"http://127.0.0.1:{self.port}/{filename}"


def wait_for_server(url: str, timeout: float = 10.0) -> bool:
    """Wait for HTTP server to respond."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1.0)
            if 200 <= response.status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.3)
    return False


async def render_ortho_png(ply_path: str, 
                          output_path: str,
                          width: int = 4096, 
                          height: int = 4096,
                          headless: bool = True,
                          timeout: float = 180.0) -> str:
    """
    Render orthographic view of PLY file to PNG.
    
    Args:
        ply_path: Path to PLY file
        output_path: Output PNG file path  
        width: Image width in pixels (default: 4096)
        height: Image height in pixels (default: 4096)
        headless: Run browser in headless mode (default: True)
        timeout: Timeout in seconds (default: 180)
        
    Returns:
        Path to output PNG file
    """
    ply_path = Path(ply_path).resolve()
    output_path = Path(output_path).resolve()
    
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Start PLY server
    server = PLYServer(str(ply_path.parent))
    server_url = server.start()
    
    browser = None
    try:
        # Wait for server
        ply_url = server.get_file_url(ply_path.name)
        if not wait_for_server(ply_url, timeout=10.0):
            raise RuntimeError("Failed to start PLY server")
        
        # Launch browser
        playwright = await async_playwright().start()
        
        args = [
            f"--window-size={width},{height}",
            "--no-first-run",
            "--no-default-browser-check", 
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--disable-gpu-sandbox",
            "--enable-accelerated-2d-canvas",
            "--use-angle=metal",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-features=MediaRouter",
            "--load-media-router-component-extension=0"
        ]
        
        browser = await playwright.chromium.launch(headless=headless, args=args)
        page = await browser.new_page()
        await page.set_viewport_size({"width": width, "height": height})
        page.set_default_timeout(timeout * 1000)
        
        # Navigate to SuperSplat
        editor_url = f"https://superspl.at/editor?load={quote(ply_url)}"
        await page.goto(editor_url, wait_until="domcontentloaded", timeout=0)
        
        # Wait for scene to load
        await page.wait_for_function("""
            () => {
                const w = window;
                try {
                    if (!w.scene || !w.scene.bound) return false;
                    const b = w.scene.bound;
                    return !!(b && b.halfExtents && (b.halfExtents.x > 0 || b.halfExtents.y > 0 || b.halfExtents.z > 0));
                } catch { 
                    return false; 
                }
            }
        """, timeout=0)
        
        # Small delay for resources
        await page.evaluate("() => new Promise(r => setTimeout(r, 1500))")
        
        # Render orthographic view
        render_script = f"""
        async ({{ width, height }}) => {{
            // Configure SuperSplat for orthographic rendering
            window.scene.events.fire('view.setBands', 3);
            window.scene.events.fire('camera.setOverlay', false);
            window.scene.events.fire('camera.setMode', 'rings');
            
            if (window.scene.events.invoke('grid.visible')) {{
                window.scene.events.fire('grid.setVisible', false);
            }}
            
            window.scene.events.fire('camera.setBound', false);
            window.scene.events.fire('view.setOutlineSelection', false);
            
            // Set black background
            try {{
                const Color = window.pc && window.pc.Color ? window.pc.Color : (window.Color || null);
                if (Color) window.scene.events.fire('setBgClr', new Color(0, 0, 0, 1));
                else window.scene.events.fire('setBgClr', {{ r: 0, g: 0, b: 0, a: 1 }});
            }} catch (e) {{}}

            // Select first splat
            const splats = window.scene.getElementsByType(window.ElementType?.splat ?? 2);
            if (splats && splats.length) {{
                window.scene.events.fire('selection', splats[0]);
            }}

            await new Promise(r => setTimeout(r, 5000));

            // Set up orthographic camera
            const b = window.scene.bound;
            const center = b.center;
            const hx = b.halfExtents.x, hy = b.halfExtents.y, hz = b.halfExtents.z;
            const radius = Math.sqrt(hx*hx + hy*hy + hz*hz);

            window.scene.events.fire('camera.align', 'pz');
            window.scene.camera.ortho = true;
            window.scene.events.fire('camera.focus');
            window.scene.camera.focus({{ focalPoint: center, radius, speed: 0 }});
            
            const aspect = width / height;
            const orthoHeight = Math.max(hy, hx / aspect) * 1.05;
            const cam = window.scene.camera.entity.camera;
            cam.orthoHeight = orthoHeight;
            cam.nearClip = -radius * 4;
            cam.farClip = radius * 4;

            window.scene.camera.focus({{ focalPoint: center, radius, speed: 0 }});
            window.scene.forceRender = true;

            // Render offscreen
            window.scene.camera.startOffscreenMode(width, height);
            window.scene.camera.renderOverlays = false;
            window.scene.forceRender = true;

            const waitFrame = () => new Promise((resolve) => {{
                const h = window.scene.events.on('postrender', () => {{ h.off(); resolve(null); }});
                setTimeout(() => {{ try {{ h.off(); }} catch {{}} resolve(null); }}, 200);
            }});
            
            // Wait for content to render
            let contentOk = false;
            for (let i = 0; i < 120 && !contentOk; i++) {{
                await waitFrame();
                const rt = window.scene.camera.entity.camera.renderTarget;
                const cb = rt.colorBuffer;
                const w = cb.width, h = cb.height;
                const data = new Uint8Array(w * h * 4);
                await cb.read(0, 0, w, h, {{ renderTarget: rt, data }});
                
                let nonBlack = 0;
                for (let p = 0; p < data.length; p += 16) {{
                    if (data[p] | data[p + 1] | data[p + 2]) {{ 
                        nonBlack++; 
                        if (nonBlack > 10) break; 
                    }}
                }}
                contentOk = nonBlack > 10;
            }}

            // Capture final image
            const rt = window.scene.camera.entity.camera.renderTarget;
            const cb = rt.colorBuffer;
            const w = cb.width, h = cb.height;
            const data = new Uint8Array(w * h * 4);
            await cb.read(0, 0, w, h, {{ renderTarget: rt, data }});

            // Create canvas and flip vertically
            const cnv = document.createElement('canvas');
            cnv.width = w; cnv.height = h;
            const ctx = cnv.getContext('2d');
            
            ctx.fillStyle = '#000000';
            ctx.fillRect(0, 0, w, h);

            const row = new Uint8ClampedArray(w * 4);
            const pixels = new Uint8ClampedArray(data.buffer);
            
            for (let y = 0; y < Math.floor(h / 2); y++) {{
                const top = y * w * 4;
                const bottom = (h - y - 1) * w * 4;
                row.set(pixels.subarray(top, top + w * 4));
                pixels.copyWithin(top, bottom, bottom + w * 4);
                pixels.set(row, bottom);
            }}
            
            for (let i = 0; i < pixels.length; i += 4) pixels[i + 3] = 255;
            const imageData = new ImageData(pixels, w, h);
            ctx.putImageData(imageData, 0, 0);

            // Cleanup
            window.scene.camera.endOffscreenMode();
            window.scene.camera.renderOverlays = true;

            return cnv.toDataURL('image/png');
        }}
        """
        
        data_url = await page.evaluate(render_script, {
            "width": width, 
            "height": height
        })
        
        # Save image
        base64_data = data_url.replace('data:image/png;base64,', '')
        image_data = base64.b64decode(base64_data)
        
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        print(f"Wrote {output_path} ({width}x{height})")
        return str(output_path)
        
    finally:
        if browser:
            await browser.close()
        server.stop()


def render_ortho(ply_path: str, 
                output_path: str,
                width: int = 4096, 
                height: int = 4096,
                headless: bool = True,
                timeout: float = 180.0) -> str:
    """
    Synchronous wrapper for render_ortho_png.
    
    Args:
        ply_path: Path to PLY file
        output_path: Output PNG file path
        width: Image width in pixels (default: 4096)
        height: Image height in pixels (default: 4096)
        headless: Run browser in headless mode (default: True)
        timeout: Timeout in seconds (default: 180)
        
    Returns:
        Path to output PNG file
    """
    return asyncio.run(render_ortho_png(
        ply_path=ply_path,
        output_path=output_path,
        width=width,
        height=height,
        headless=headless,
        timeout=timeout
    ))


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python splats_ortho.py input.ply output.png [width] [height]")
        sys.exit(1)
    
    ply_file = sys.argv[1]
    output_file = sys.argv[2]
    width = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    height = int(sys.argv[4]) if len(sys.argv) > 4 else width
    
    try:
        render_ortho(ply_file, output_file, width, height)
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

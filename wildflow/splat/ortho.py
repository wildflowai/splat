"""
Orthographic PNG rendering from PLY files using SuperSplat.
"""

import asyncio
import base64
import os
import shutil
import socket
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import requests
from PIL import Image
from tqdm.asyncio import tqdm

# Constants
BLACK_THRESHOLD_RATIO = 0.95
BUFFER_PIXELS = 20
CAMERA_Z_POSITION_METERS = 200.0
CELL_BUFFER_METERS = 0.2
DARK_PIXEL_THRESHOLD = 10
MAX_CONCURRENT_RENDERS = 4
MAX_RETRIES_COUNT = 3
RENDER_DELAY_MILLISECONDS = 1500
SAFETY_MARGIN_METERS = 0.1
SERVER_TIMEOUT_SECONDS = 10.0
SUPERSPLAT_URL = "https://superspl.at/editor"
TIMEOUT_SECONDS = 180.0

Image.MAX_IMAGE_PIXELS = None
os.environ["OBJC_DISABLE_INITIALIZE_FOR_EXTENSIONS"] = "1"


class _CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, serve_directory: str = None, **kwargs):
        self.serve_directory = serve_directory or os.getcwd()
        super().__init__(*args, directory=self.serve_directory, **kwargs)

    def do_GET(self):
        try:
            super().do_GET()
        except BrokenPipeError:
            # This happens when the browser disconnects before the full file is sent.
            # It's safe to ignore as the browser has what it needs.
            pass

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def guess_type(self, path):
        if path.endswith(".ply"):
            return "application/ply"
        return super().guess_type(path)

    def log_message(self, format, *args):
        pass


class _PLYServer:
    def __init__(self, serve_directory: str):
        self.serve_directory = serve_directory
        self.server = None
        self.thread = None
        self.port = None

    def _find_free_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            return s.getsockname()[1]

    def start(self):
        self.port = self._find_free_port()

        def run_server():
            handler = lambda *args, **kwargs: _CORSHTTPRequestHandler(
                *args, serve_directory=self.serve_directory, **kwargs
            )
            self.server = HTTPServer(("localhost", self.port), handler)
            self.server.serve_forever()

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

        return f"http://localhost:{self.port}"

    def get_file_url(self, filename: str):
        return f"http://localhost:{self.port}/{quote(filename)}"

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()


def _wait_for_server(url: str, timeout: float = SERVER_TIMEOUT_SECONDS) -> bool:
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            return response.status_code == 200
        except:
            time.sleep(0.1)
    return False


def _generate_grid_cells(
    min_x: float, max_x: float, min_y: float, max_y: float, cell_size: float
) -> List[Tuple[float, float, int, int]]:
    width = max_x - min_x
    height = max_y - min_y

    cols = int(np.ceil(width / cell_size))
    rows = int(np.ceil(height / cell_size))

    cells = []
    for row in range(rows):
        for col in range(cols):
            center_x = min_x + (col + 0.5) * cell_size
            center_y = min_y + (row + 0.5) * cell_size
            cells.append((center_x, center_y, col, row))

    return cells


async def _render_cell(
    ply_path: str,
    output_path: str,
    width: int,
    height: int,
    meters_per_pixel: float,
    center: Optional[tuple[float, float, float]] = None,
) -> str:
    ply_path = Path(ply_path).resolve()
    output_path = Path(output_path).resolve()

    assert ply_path.exists(), f"PLY file not found: {ply_path}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_js = Path(__file__).parent / "ortho_renderer.js"
    assert render_js.exists(), f"ortho_renderer.js not found: {render_js}"

    with open(render_js, "r") as f:
        render_script = f.read()

    server = _PLYServer(str(ply_path.parent))
    server_url = server.start()

    browser = None
    try:
        ply_url = server.get_file_url(ply_path.name)
        assert _wait_for_server(
            ply_url, timeout=SERVER_TIMEOUT_SECONDS), "Failed to start PLY server"

        from playwright.async_api import async_playwright

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
            "--load-media-router-component-extension=0",
        ]

        browser = await playwright.chromium.launch(headless=True, args=args)
        page = await browser.new_page()
        await page.set_viewport_size({"width": width, "height": height})
        page.set_default_timeout(TIMEOUT_SECONDS * 1000)

        editor_url = f"{SUPERSPLAT_URL}?load={quote(ply_url)}"
        await page.goto(editor_url, wait_until="domcontentloaded", timeout=0)

        await page.wait_for_function(
            """
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
        """,
            timeout=0,
        )

        await page.evaluate(f"() => new Promise(r => setTimeout(r, {RENDER_DELAY_MILLISECONDS}))")

        camera_center = None
        if center:
            camera_center = {"x": -center[0], "y": -center[1], "z": center[2]}

        data_url = await page.evaluate(
            render_script,
            {"width": width, "height": height,
                "metersPerPixel": meters_per_pixel, "cameraCenter": camera_center},
        )

        assert data_url.startswith(
            "data:image/png;base64,"), "Invalid PNG data returned"

        png_data = base64.b64decode(data_url.split(",")[1])

        with open(output_path, "wb") as f:
            f.write(png_data)

        return str(output_path)

    finally:
        if browser:
            await browser.close()
        server.stop()


def _stitch_and_crop(
    cell_files: List[Path],
    cols: int,
    rows: int,
    cell_pixels: int,
    output_path: Path,
    crop_bounds: Tuple[float, float, float, float],
    cell_size: float,
    meters_per_pixel: float,
):
    min_x, max_x, min_y, max_y = crop_bounds

    ortho_width = cols * cell_pixels
    ortho_height = rows * cell_pixels

    orthomosaic = Image.new("RGB", (ortho_width, ortho_height), (0, 0, 0))

    placed_cells = 0
    for cell_file in cell_files:
        parts = cell_file.stem.split("_")
        col = int(parts[1])
        row = int(parts[2])

        cell_img = Image.open(cell_file)
        assert cell_img.size == (
            cell_pixels, cell_pixels), f"Cell size mismatch: {cell_img.size}"

        x = col * cell_pixels
        y = (rows - 1 - row) * cell_pixels

        orthomosaic.paste(cell_img, (x, y))
        placed_cells += 1

    grid_max_x = min_x + cols * cell_size
    grid_max_y = min_y + rows * cell_size

    padding_right = grid_max_x - max_x
    padding_top = grid_max_y - max_y + SAFETY_MARGIN_METERS

    padding_right_px = int(padding_right / meters_per_pixel)
    padding_top_px = int(padding_top / meters_per_pixel)

    crop_left = 0
    crop_top = padding_top_px
    crop_right = orthomosaic.width - padding_right_px
    crop_bottom = orthomosaic.height

    crop_left = max(0, crop_left)
    crop_right = max(crop_left + 100, min(orthomosaic.width, crop_right))
    crop_top = max(0, crop_top)
    crop_bottom = max(crop_top + 100, min(orthomosaic.height, crop_bottom))

    orthomosaic = orthomosaic.crop(
        (crop_left, crop_top, crop_right, crop_bottom))
    orthomosaic.save(output_path, "PNG", quality=95)


def _is_image_mostly_black(image_path: Path, black_threshold: float = BLACK_THRESHOLD_RATIO) -> bool:
    try:
        img = Image.open(image_path).convert("L")
        pixels = list(img.getdata())
        black_pixels = sum(1 for pixel in pixels if pixel <
                           DARK_PIXEL_THRESHOLD)
        black_ratio = black_pixels / len(pixels)
        return black_ratio > black_threshold
    except Exception:
        return True


async def _render_and_process_cell(
    semaphore: asyncio.Semaphore,
    cell_data: tuple,
    tmp_dir: Path,
    render_size: int,
    target_pixels: int,
    meters_per_pixel: float,
    progress_callback=None,
) -> Optional[Path]:
    center_x, center_y, col, row = cell_data
    cell_png_file = tmp_dir / f"cell_{col:03d}_{row:03d}.png"
    cell_ply_file = tmp_dir / f"cell_{col:03d}_{row:03d}.ply"

    if not cell_ply_file.exists():
        return None

    max_retries = MAX_RETRIES_COUNT

    try:
        for attempt in range(max_retries):
            async with semaphore:
                await _render_cell(
                    str(cell_ply_file),
                    str(cell_png_file),
                    render_size,
                    render_size,
                    meters_per_pixel,
                    (center_x, center_y, 0.0),
                )

            if not cell_png_file.exists():
                if attempt == max_retries - 1:
                    return None
                continue

            # Check if the rendered image is mostly black
            if _is_image_mostly_black(cell_png_file):
                if attempt < max_retries - 1:
                    print(
                        f"Cell {col:03d}_{row:03d} rendered black, retrying... (attempt {attempt + 2}/{max_retries})")
                    if cell_png_file.exists():
                        cell_png_file.unlink()
                    await asyncio.sleep(1)
                    continue
                else:
                    print(
                        f"Cell {col:03d}_{row:03d} still black after {max_retries} attempts, keeping it")

            # Successful render, process the image
            break

        if progress_callback:
            await progress_callback()

        if not cell_png_file.exists():
            return None

        img = Image.open(cell_png_file)
        flipped_img = img.rotate(180)

        crop_margin = BUFFER_PIXELS
        crop_box = (crop_margin, crop_margin, render_size -
                    crop_margin, render_size - crop_margin)
        cropped_img = flipped_img.crop(crop_box)

        assert cropped_img.size == (
            target_pixels, target_pixels), f"Size mismatch: {cropped_img.size}"
        cropped_img.save(cell_png_file)

        if progress_callback:
            await progress_callback()

        return cell_png_file

    finally:
        if cell_ply_file.exists():
            cell_ply_file.unlink()


async def _ortho_async(input_ply: str, output_png: str, meters_per_pixel: float, cell_size_meters: float) -> None:
    """Internal async implementation of ortho function."""
    from ._core import OrthoCells

    input_path = Path(input_ply)
    output_path = Path(output_png)

    assert input_path.exists(), f"PLY file not found: {input_path}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load PLY and get bounds
    ortho_cells = OrthoCells(str(input_path))
    min_x, max_x, min_y, max_y, avg_z = ortho_cells.get_bounds()

    # Create temporary directory for processing
    tmp_dir = Path("tmp_ortho")
    tmp_dir.mkdir(exist_ok=True)

    try:
        print(
            f"PLY bounds: x=[{min_x:.3f}, {max_x:.3f}], y=[{min_y:.3f}, {max_y:.3f}]")

        cells = _generate_grid_cells(
            min_x, max_x, min_y, max_y, cell_size_meters)

        cols = int(np.ceil((max_x - min_x) / cell_size_meters))
        rows = int(np.ceil((max_y - min_y) / cell_size_meters))

        target_pixels = int(cell_size_meters / meters_per_pixel)
        render_size = target_pixels + (2 * BUFFER_PIXELS)

        print("Preparing cell definitions...")
        cell_definitions = []
        for center_x, center_y, col, row in cells:
            half_cell = cell_size_meters / 2
            buffer = CELL_BUFFER_METERS
            min_x_b = center_x - half_cell - buffer
            max_x_b = center_x + half_cell + buffer
            min_y_b = center_y - half_cell - buffer
            max_y_b = center_y + half_cell + buffer
            cell_ply_path = str(tmp_dir / f"cell_{col:03d}_{row:03d}.ply")
            cell_definitions.append(
                (min_x_b, max_x_b, min_y_b, max_y_b, cell_ply_path))

        print(f"Extracting {len(cell_definitions)} cells...")
        ortho_cells.extract_all_cells(cell_definitions)

        # Render cells in parallel
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_RENDERS)

        rendered_cells = []
        total_steps = len(cells) * 2
        completed_steps = 0

        with tqdm(total=100, desc="Rendering & Processing", unit="%") as pbar:

            async def progress_update():
                nonlocal completed_steps
                completed_steps += 1
                progress = int((completed_steps / total_steps) * 90)
                pbar.update(progress - pbar.n)

            tasks = [
                _render_and_process_cell(
                    semaphore, cell_data, tmp_dir, render_size, target_pixels, meters_per_pixel, progress_update
                )
                for cell_data in cells
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)
            rendered_cells = [
                r for r in results if isinstance(r, Path) and r.exists()]

            errors = [r for r in results if isinstance(r, Exception)]
            if errors:
                print(f"Encountered {len(errors)} errors during rendering.")

            pbar.set_description("Stitching orthomosaic")
            pbar.update(90 - pbar.n)

            rendered_cells.sort()
            print(f"Stitching {len(rendered_cells)} rendered cells...")
            if not rendered_cells:
                print("ERROR: No cells were rendered successfully!")
                return

            _stitch_and_crop(
                rendered_cells,
                cols,
                rows,
                target_pixels,
                output_path,
                (min_x, max_x, min_y, max_y),
                cell_size_meters,
                meters_per_pixel,
            )
            print(f"Orthomosaic saved to: {output_path}")

            pbar.update(100 - pbar.n)
            pbar.set_description("Complete")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

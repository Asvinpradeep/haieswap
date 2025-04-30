from flask import Flask, request, jsonify
from gradio_client import Client, file
import requests
import os
import traceback
import concurrent.futures
import psutil
import signal
from PIL import Image
from io import BytesIO

app = Flask(__name__)
client = Client("AIRI-Institute/HairFastGAN")
TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/upload"
MEMORY_THRESHOLD_MB = 900  # threshold in megabytes to force a restart


def ensure_memory_or_restart():
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    if rss > MEMORY_THRESHOLD_MB:
        print(f"[MEMORY] {rss:.0f}MB > {MEMORY_THRESHOLD_MB}MB — killing process to force a cold start")
        os.kill(os.getpid(), signal.SIGKILL)


def download_resize_upload(url):
    """
    Download an image, resize it to 480x480, upload it to tmpfiles.org, and return the new URL.
    """
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    img_resized = img.resize((480, 480))

    temp_filename = "temp_resized.jpg"
    img_resized.save(temp_filename)

    with open(temp_filename, 'rb') as f:
        files = {'file': (os.path.basename(temp_filename), f)}
        resp = requests.post(TMPFILES_UPLOAD_URL, files=files)

    if resp.status_code != 200:
        raise Exception(f"Upload failed: {resp.status_code}: {resp.text}")

    try:
        result = resp.json()
        uploaded_url = (result.get('data', {}).get('url') or
                        result.get('url') or
                        result.get('link') or
                        result)
    except ValueError:
        uploaded_url = resp.text.strip()

    if isinstance(uploaded_url, str) and uploaded_url.startswith("https://tmpfiles.org/") and "/dl/" not in uploaded_url:
        uploaded_url = uploaded_url.replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/")

    os.remove(temp_filename)
    return uploaded_url


def upload_local_file(local_path):
    """
    Upload a local file to tmpfiles.org and return the direct-download URL.
    """
    with open(local_path, 'rb') as f:
        files = {'file': (os.path.basename(local_path), f)}
        resp = requests.post(TMPFILES_UPLOAD_URL, files=files)

    if resp.status_code != 200:
        raise Exception(f"Upload failed: {resp.status_code}: {resp.text}")

    try:
        result = resp.json()
        url = (result.get('data', {}).get('url') or
               result.get('url') or
               result.get('link') or
               result)
    except ValueError:
        url = resp.text.strip()

    if isinstance(url, str) and url.startswith("https://tmpfiles.org/") and "/dl/" not in url:
        url = url.replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/")

    return url


@app.route('/process-hair-swap', methods=['POST'])
def process_hair_swap():
    ensure_memory_or_restart()

    data = request.get_json(force=True)

    face_url = data.get('face_url')
    shape_url = data.get('shape_url')
    color_url = data.get('color_url')
    if not all([face_url, shape_url, color_url]):
        return jsonify({"error": "face_url, shape_url, and color_url are required"}), 400

    local_files = []
    try:
        def process_image(url, api_name):
            ensure_memory_or_restart()

            # Step 1: download, resize, upload
            resized_url = download_resize_upload(url)

            # Step 2: Call Gradio with resized URL
            output_path = client.predict(
                img=file(resized_url),
                align=["Face", "Shape", "Color"],
                api_name=api_name
            )
            local_files.append(output_path)

            # Step 3: Upload output
            return upload_local_file(output_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_face = executor.submit(process_image, face_url, "/resize_inner")
            future_shape = executor.submit(process_image, shape_url, "/resize_inner_1")
            future_color = executor.submit(process_image, color_url, "/resize_inner_2")
            face_dl_url = future_face.result()
            shape_dl_url = future_shape.result()
            color_dl_url = future_color.result()

        ensure_memory_or_restart()

        # Now swap hair
        swap_output = client.predict(
            face=file(face_dl_url),
            shape=file(shape_dl_url),
            color=file(color_dl_url),
            blending=data.get('blending', "Article"),
            poisson_iters=int(data.get('poisson_iters', 2500)),
            poisson_erosion=int(data.get('poisson_erosion', 100)),
            api_name="/swap_hair"
        )

        if isinstance(swap_output, (tuple, list)):
            swapped_local = next(
                (item['value'] for item in swap_output if isinstance(item, dict) and item.get('visible') and 'value' in item),
                None
            )
            if not swapped_local:
                raise Exception(f"Unexpected swap output format: {swap_output}")
        else:
            swapped_local = swap_output

        local_files.append(swapped_local)

        swapped_dl_url = upload_local_file(swapped_local)
        return jsonify({"result_url": swapped_dl_url}), 200

    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean temp files
        for path in local_files:
            try:
                if path and os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
from gradio_client import Client, file
from PIL import Image
from io import BytesIO
import requests
import os
import traceback
import psutil
import signal

app = Flask(__name__)
client = Client("AIRI-Institute/HairFastGAN")
TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/upload"
MEMORY_THRESHOLD_MB = 900  # in megabytes


def ensure_memory_or_restart():
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    if rss > MEMORY_THRESHOLD_MB:
        print(f"[MEMORY] {rss:.0f}MB > {MEMORY_THRESHOLD_MB}MB — killing process to force restart")
        os.kill(os.getpid(), signal.SIGKILL)


def resize_and_upload(url):
    response = requests.get(url)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content)).convert("RGB")
    img_resized = img.resize((480, 480))

    img_byte_arr = BytesIO()
    img_resized.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    files = {'file': ('resized.jpg', img_byte_arr, 'image/jpeg')}
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

    temp_urls = []
    try:
        # Resize and upload
        face_resized_url = resize_and_upload(face_url)
        shape_resized_url = resize_and_upload(shape_url)
        color_resized_url = resize_and_upload(color_url)
        temp_urls.extend([face_resized_url, shape_resized_url, color_resized_url])

        # Predict hair swap
        ensure_memory_or_restart()
        swap_output = client.predict(
            face=file(face_resized_url),
            shape=file(shape_resized_url),
            color=file(color_resized_url),
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

        # Re-upload final output if needed
        swapped_final_url = resize_and_upload(swapped_local)
        temp_urls.append(swapped_final_url)

        return jsonify({"result_url": swapped_final_url}), 200

    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # No local files to delete — only memory
        temp_urls.clear()  # clear in-memory URL list to free memory

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

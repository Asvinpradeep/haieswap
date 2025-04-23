from flask import Flask, request, jsonify
from gradio_client import Client, file
import requests
import os
import traceback
import concurrent.futures
import psutil
import signal

app = Flask(__name__)
client = Client("AIRI-Institute/HairFastGAN")
TMPFILES_UPLOAD_URL = "https://tmpfiles.org/api/v1/upload"
MEMORY_THRESHOLD_MB = 900  # threshold in megabytes to force a restart

def ensure_memory_or_restart():
    """
    Kill the process if Resident Set Size (RSS) exceeds the threshold,
    causing Vercel to spin up a fresh container on the next request.
    """
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    if rss > MEMORY_THRESHOLD_MB:
        print(f"[MEMORY] {rss:.0f}MB > {MEMORY_THRESHOLD_MB}MB — killing process to force a cold start")
        os.kill(os.getpid(), signal.SIGKILL)


def upload_to_tmpfiles(local_path):
    """
    Upload a local file to tmpfiles.org and return a direct-download URL.
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
    # Force a cold start if memory is too high
    ensure_memory_or_restart()

    data = request.get_json(force=True)

    face_url = data.get('face_url')
    shape_url = data.get('shape_url')
    color_url = data.get('color_url')
    if not all([face_url, shape_url, color_url]):
        return jsonify({"error": "face_url, shape_url, and color_url are required"}), 400

    local_files = []
    try:
        # Parallel resize and upload for face, shape, color
        def resize_and_dl(url, api_name):
            ensure_memory_or_restart()
            local_path = client.predict(
                img=file(url),
                align=["Face", "Shape", "Color"],
                api_name=api_name
            )
            local_files.append(local_path)
            return upload_to_tmpfiles(local_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_face = executor.submit(resize_and_dl, face_url, "/resize_inner")
            future_shape = executor.submit(resize_and_dl, shape_url, "/resize_inner_1")
            future_color = executor.submit(resize_and_dl, color_url, "/resize_inner_2")
            face_dl_url = future_face.result()
            shape_dl_url = future_shape.result()
            color_dl_url = future_color.result()

        # Perform hair swap
        ensure_memory_or_restart()
        swap_output = client.predict(
            face=file(face_dl_url),
            shape=file(shape_dl_url),
            color=file(color_dl_url),
            blending=data.get('blending', "Article"),
            poisson_iters=int(data.get('poisson_iters', 2500)),
            poisson_erosion=int(data.get('poisson_erosion', 100)),
            api_name="/swap_hair"
        )

        # Unpack tuple output if needed
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
        swapped_dl_url = upload_to_tmpfiles(swapped_local)
        return jsonify({"result_url": swapped_dl_url}), 200

    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up all temporary files
        for path in local_files:
            try:
                if path and os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

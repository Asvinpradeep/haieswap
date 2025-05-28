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

IMGBB_API_KEY = "9055cb00c5a5912a2aa30ec07701a757"  # Store securely in production
IMGBB_UPLOAD_URL = f"https://api.imgbb.com/1/upload?expiration=600&key={IMGBB_API_KEY}"
MEMORY_THRESHOLD_MB = 900

def ensure_memory_or_restart():
    """
    Kill the process if memory usage exceeds threshold to force a cold restart (for Vercel or similar).
    """
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    if rss > MEMORY_THRESHOLD_MB:
        print(f"[MEMORY] {rss:.0f}MB > {MEMORY_THRESHOLD_MB}MB — killing process to force a cold start")
        os.kill(os.getpid(), signal.SIGKILL)

def upload_to_imgbb(local_path):
    """
    Upload a local image file to ImgBB and return the direct image URL.
    """
    with open(local_path, "rb") as f:
        files = {"image": f}
        resp = requests.post(IMGBB_UPLOAD_URL, files=files)

    if resp.status_code != 200:
        raise Exception(f"ImgBB upload failed: {resp.status_code}: {resp.text}")

    data = resp.json()
    if not data.get("success"):
        raise Exception("ImgBB upload failed: response indicates failure")

    return data["data"]["url"]

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
        # Parallel resize and upload for face, shape, color
        def resize_and_upload(url, api_name):
            ensure_memory_or_restart()
            local_path = client.predict(
                img=file(url),
                align=["Face", "Shape", "Color"],
                api_name=api_name
            )
            local_files.append(local_path)
            return upload_to_imgbb(local_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_face = executor.submit(resize_and_upload, face_url, "/resize_inner")
            future_shape = executor.submit(resize_and_upload, shape_url, "/resize_inner_1")
            future_color = executor.submit(resize_and_upload, color_url, "/resize_inner_2")
            face_dl_url = future_face.result()
            shape_dl_url = future_shape.result()
            color_dl_url = future_color.result()

        ensure_memory_or_restart()
        swap_output = client.predict(
            face=file(face_dl_url),
            shape=file(shape_dl_url),
            color=file(color_dl_url),
            blending=data.get('blending', "Article"),
            poisson_iters=int(data.get('poisson_iters', 0)),
            poisson_erosion=int(data.get('poisson_erosion', 15)),
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
        swapped_dl_url = upload_to_imgbb(swapped_local)
        return jsonify({"result_url": swapped_dl_url}), 200

    except Exception as e:
        print("Exception occurred:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        for path in local_files:
            try:
                if path and os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

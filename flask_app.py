from flask import Flask, request, jsonify, send_file
import subprocess
import tempfile
import os

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/infer', methods=['POST'])
def run_inference():
    try:
        audio_file = request.files['driven_audio']
        image_file = request.files['source_image']
        enhancer = request.form.get('enhancer', 'gfpgan')

        if not audio_file or not image_file:
            return jsonify({'error': 'Both driven_audio and source_image files are required.'}), 400

        if not allowed_file(image_file.filename):
            return jsonify({'error': 'Invalid file format for source_image.'}), 400

        # Save the audio file
        audio_file.save('driven_audio.wav')

        # Create a temporary directory to store the image file
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, 'source_image' + os.path.splitext(image_file.filename)[1])
            image_file.save(image_path)

            # Run the inference.py script
            output_dir = tempfile.mkdtemp()
            cmd = [
                'python3',
                'inference.py',
                '--driven_audio', 'driven_audio.wav',
                '--source_image', image_path,
                '--enhancer', enhancer,
                '--result_dir', output_dir
            ]
            subprocess.run(cmd)

            # List files in the output directory
            output_files = os.listdir(output_dir)

            # Assuming there is only one video file in the output directory
            for output_file in output_files:
                if output_file.endswith('.mp4'):
                    output_video_path = os.path.join(output_dir, output_file)
                    return send_file(output_video_path, mimetype='video/mp4')

            return jsonify({'error': 'No video file found in the output directory.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/', methods=['GET'])
def check_status():
    return 'App is running.'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


import logging
import os
import re
import sys
import threading
import warnings
import tempfile
from stslib import cfg, tool
from faster_whisper import WhisperModel

warnings.filterwarnings('ignore')


def configure_logging():
    log = logging.getLogger()
    log.handlers = []
    log.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)
    return log


logger = configure_logging()


def process_file(input_file, model_name, language):
    wav_file = None

    try:
        sets = cfg.parse_ini()
        model = WhisperModel(
            model_name.replace('-whisper', '') if model_name.startswith('distil') else model_name,
            device=sets.get('devtype'),
            compute_type=sets.get('cuda_com_type'),
            download_root=cfg.ROOT_DIR + "/models",
            local_files_only=False if '/' in model_name else True
        )

        logger.info(f"Processing file: {input_file}")

        wav_file = convert_to_wav(input_file)

        segments, info = model.transcribe(
            wav_file,
            beam_size=sets.get('beam_size'),
            best_of=sets.get('best_of'),
            condition_on_previous_text=sets.get('condition_on_previous_text'),
            vad_filter=sets.get('vad'),
            language=language if language != 'auto' else None,
            initial_prompt=sets.get('initial_prompt_zh') if language == 'zh' else None
        )

        write_srt(segments, input_file, info.duration)

    except Exception as e:
        logger.error(f"Error processing {input_file}: {str(e)}")

    finally:
        if wav_file and os.path.exists(wav_file):
            os.remove(wav_file)
            logger.info(f"Deleted temporary file: {wav_file}")


def convert_to_wav(input_file):
    """Converts various audio/video file types to wav format in a temporary directory"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
        wav_file_path = temp_wav_file.name

    basename, _ = os.path.splitext(os.path.basename(input_file))
    wav_file = os.path.join(cfg.TMP_DIR, f'{basename}.wav')
    params = [
        "-i", input_file,
        "-ar", "16000",
        "-ac", "1",
        "-af", "aresample=async=1",
        wav_file_path
    ]
    rs = tool.runffmpeg(params)
    if rs != 'ok':
        raise RuntimeError(f"FFmpeg error: {rs}")

    return wav_file_path


def write_srt(segments, original_filename, total_duration):
    """Writes the segments to an SRT file with the same basename as the original file."""
    srt_content = []
    for segment in segments:
        progress = segment.end / total_duration
        logger.info(f"Progress: {progress*100:.2f}% for file: {original_filename}")

        startTime = tool.ms_to_time_string(ms=int(segment.start * 1000))
        endTime = tool.ms_to_time_string(ms=int(segment.end * 1000))
        text = segment.text.strip().replace('&#39;', "'")
        text = re.sub(r'&#\d+;', '', text)

        if text and not re.match(r'^[，。、？‘’“”；：（｛｝【】）:;"\'\s \d`!@#$%^&*()_+=.,?/\\-]*$', text):
            srt_content.append(f'{len(srt_content) + 1}\n{startTime} --> {endTime}\n{text}\n')

    if srt_content:
        srt_file = os.path.splitext(original_filename)[0] + ".srt"
        with open(srt_file, 'w', encoding='utf-8') as file:
            file.write("\n".join(srt_content))

        logger.info(f"Subtitle saved to: {srt_file}")
    else:
        logger.warning(f"No valid segments found for {original_filename}")


def main(args):
    if len(args) < 3:
        print("Usage: python script.py <model_name> <language> <input_files...>")
        sys.exit(1)

    model_name = args[0]
    language = args[1]
    input_files = args[2:]

    for input_file in input_files:
        process_file(input_file, model_name, language)

    # threads = []
    # for input_file in input_files:
    #     thread = threading.Thread(target=process_file, args=(input_file, model_name, language))
    #     thread.start()
    #     threads.append(thread)
    #
    # for thread in threads:
    #     thread.join()


if __name__ == "__main__":
    main(sys.argv[1:])

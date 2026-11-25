import json
import logging
import os

from dotenv import load_dotenv
from google import genai

from log_config import setup_logging
from search import cari_resep

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
MAX_TOOL_ROUNDS = 5  # safety net against an endless tool-call loop
MAX_RESULTS = 10  # upper bound for `jumlah`, keeps the tokens sent back to the LLM in check

logger = logging.getLogger("agent")

SYSTEM_INSTRUCTION = (
    "Kamu asisten resep. Jawab dalam bahasa Indonesia. "
    "Jawab HANYA berdasarkan hasil tool cari_resep. Kalau hasil pencarian tidak memuat "
    "jawabannya, katakan bahwa resep itu tidak ada di koleksi; jangan memakai pengetahuan lain. "
    "Untuk sapaan atau ucapan terima kasih, balas singkat tanpa mencari. "
    "Sebutkan judul resep yang kamu pakai. "
    "Setiap permintaan dengan kriteria baru, panggil cari_resep lagi dengan SEMUA kriteria digabung, "
    "termasuk topik dari percakapan sebelumnya kalau user merujuk ke 'tadi' atau 'list kamu'. "
    "Jangan menjawab ketersediaan resep dari ingatan hasil sebelumnya. "
    "kata_kunci wajib diisi: kalau user hanya menyebut filter, pakai topik sebelumnya, "
    "atau 'resep' kalau belum ada topik. "
    "Jangan memakai kata negasi (bukan, selain) di kata_kunci. "
    "Untuk 'selain X', cari dengan jumlah=10 lalu buang sendiri resep X dari hasil yang kamu baca. "
    "Kalau ragu topik sebelumnya masih berlaku, tanyakan ke user."
)

CARI_RESEP_TOOL = {
    "type": "function",
    "name": "cari_resep",
    "description": "Cari resep berdasarkan makna, dengan filter opsional. Mengembalikan 3 resep (bisa sampai 10 lewat jumlah).",
    "parameters": {
        "type": "object",
        "properties": {
            "kata_kunci": {"type": "string", "description": "Kata pencarian singkat, tanpa sapaan"},
            "pakai_santan": {"type": "boolean", "description": "true = hanya yang bersantan, false = hanya yang tanpa santan. Kosongkan kalau tidak disebut."},
            "vegetarian": {"type": "boolean", "description": "true = hanya vegetarian, false = hanya non-vegetarian. Kosongkan kalau tidak disebut."},
            "pedas": {"type": "boolean", "description": "true = hanya yang pedas, false = hanya yang tidak pedas. Kosongkan kalau tidak disebut."},
            "maks_menit": {"type": "integer", "description": "Waktu masak maksimal dalam menit. Kosongkan kalau tidak disebut."},
            "jumlah": {"type": "integer", "description": "Jumlah resep yang diambil, 1 sampai 10. Default 3. Naikkan kalau perlu menyaring sendiri hasilnya."},
        },
        "required": ["kata_kunci"],
    },
}


def run_tool(name: str, arguments: dict) -> list[dict] | dict:
    if name != "cari_resep":
        raise ValueError(f"unknown tool: {name}")
    arguments = dict(arguments)
    try:
        jumlah = int(arguments.pop("jumlah", 3))
        hasil = cari_resep(**arguments, limit=min(max(jumlah, 1), MAX_RESULTS))
    except ValueError as e:
        # Invalid arguments from the LLM go back to it as a result so it can correct the call.
        # MAX_TOOL_ROUNDS in answer() bounds how often it can retry.
        logger.warning("cari_resep(%s) rejected: %s", arguments, e)
        return {"error": str(e)}
    logger.info("cari_resep(%s) -> %d results", arguments, len(hasil))
    return [{"judul": r["judul"], "teks": r["teks"]} for r in hasil]


def _create(client: genai.Client, user_input, previous_id: str | None):
    # Tools and system_instruction are interaction-scoped: they must be re-sent on every call.
    kwargs = {"previous_interaction_id": previous_id} if previous_id else {}
    return client.interactions.create(
        model=MODEL,
        input=user_input,
        tools=[CARI_RESEP_TOOL],
        system_instruction=SYSTEM_INSTRUCTION,
        **kwargs,
    )


def answer(client: genai.Client, user_text: str, previous_id: str | None):
    interaction = _create(client, user_text, previous_id)
    for _ in range(MAX_TOOL_ROUNDS):
        calls = [s for s in interaction.steps if s.type == "function_call"]
        if not calls:
            return interaction
        # The LLM only *asks* for tools; our code runs them and sends the results back.
        results = [
            {
                "type": "function_result",
                "name": call.name,
                "call_id": call.id,
                "result": [{"type": "text", "text": json.dumps(run_tool(call.name, call.arguments), ensure_ascii=False)}],
            }
            for call in calls
        ]
        interaction = _create(client, results, interaction.id)
    raise RuntimeError(f"no final answer after {MAX_TOOL_ROUNDS} tool rounds")


def main() -> None:
    setup_logging(console=False)  # keep the chat clean; details go to resep_rag.log
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Check your .env file.")
    client = genai.Client(api_key=api_key)

    previous_id: str | None = None  # id of the last interaction = the conversation so far
    print("Tanya soal resep. Ketik 'exit' untuk keluar.\n")
    while True:
        try:
            user_text = input("Kamu: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_text.lower() in {"exit", "quit"}:
            break
        if not user_text:
            continue
        logger.info("user: %s", user_text)
        try:
            interaction = answer(client, user_text, previous_id)
        except Exception:
            logger.exception("turn failed")
            print("Maaf, ada error. Detailnya ada di resep_rag.log.\n")
            continue
        previous_id = interaction.id
        print(f"Bot: {interaction.output_text}\n")


if __name__ == "__main__":
    main()

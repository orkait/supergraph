import os
import json
import random
import time
import logging
import asyncio
from collections import Counter
from dataclasses import dataclass
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_DEFAULT_TRAINING_MODEL = "deepseek/deepseek-v3.2:nitro"
_TRAINING_TIMEOUT_S = 180


@dataclass(frozen=True)
class GenerationSettings:
    total_sentences_target: int
    sentences_per_call: int
    total_calls: int
    max_concurrent_calls: int
    output_file: str
    wal_file: str


@dataclass(frozen=True)
class ValidationReport:
    total: int
    valid_batch: list
    reason_counts: dict[str, int]
    invalid_examples: list[dict]

    @property
    def valid_count(self):
        return len(self.valid_batch)

    @property
    def invalid_count(self):
        return self.total - self.valid_count


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def get_generation_settings():
    total_sentences_target = _env_int("TRAINING_TOTAL_SENTENCES_TARGET", 6000)
    sentences_per_call = _env_int("TRAINING_SENTENCES_PER_CALL", 20)
    max_concurrent_calls = _env_int("TRAINING_MAX_CONCURRENT_CALLS", 8)
    output_file = os.environ.get("TRAINING_OUTPUT_FILE", "sovereign_moe_dataset.jsonl")
    wal_file = os.environ.get("TRAINING_WAL_FILE", "wal.json")
    total_calls = (total_sentences_target + sentences_per_call - 1) // sentences_per_call
    return GenerationSettings(
        total_sentences_target=total_sentences_target,
        sentences_per_call=sentences_per_call,
        total_calls=total_calls,
        max_concurrent_calls=max_concurrent_calls,
        output_file=output_file,
        wal_file=wal_file,
    )


def _preview_text(value, limit=500):
    if value is None:
        return "<none>"
    text = str(value).replace("\n", "\\n")
    if len(text) > limit:
        return f"{text[:limit]}...<truncated {len(text) - limit} chars>"
    return text


def _preview_sequence(value, limit=12):
    if not isinstance(value, list):
        return value
    if len(value) <= limit:
        return value
    return value[:limit] + [f"...<{len(value) - limit} more>"]

def load_openrouter_config():
    from tools.autoresearch.providers import load_config, resolve_providers

    config = load_config()
    providers = resolve_providers(config)
    openrouter = next((p for p in providers if p["pid"] == "openrouter"), None)
    if not openrouter:
        raise RuntimeError(
            "openrouter provider not resolved. Check tools/autoresearch/config.json "
            "(providers.openrouter) + /.env (OPENROUTER_API_KEY)."
        )

    target_model = os.environ.get("OPENROUTER_TRAINING_MODEL", _DEFAULT_TRAINING_MODEL)
    logger.info(
        "[config] provider=openrouter base_url=%s model=%s timeout=%ss",
        openrouter["api_base"], target_model, _TRAINING_TIMEOUT_S,
    )

    return {
        "api_key": openrouter["api_key"],
        "base_url": openrouter["api_base"],
        "model": target_model,
        "timeout": _TRAINING_TIMEOUT_S,
    }

ONTOLOGY = {
    "Group 1 (Software)": ["PROG_LANG", "FRAMEWORK", "API_ENDPOINT", "DATABASE", "CLOUD_SERVICE", "CLI_COMMAND", "EXCEPTION_TYPE", "IP_ADDRESS", "FILE_PATH", "SECURITY_PROTOCOL"],
    "Group 2 (Financial)": ["CURRENCY", "AMOUNT_NUM", "STOCK_TICKER", "BANK_NAME", "FINANCIAL_METRIC", "ASSET_CLASS", "INTEREST_RATE", "TAX_FORM", "MARKET_INDEX", "PAYMENT_METHOD"],
    "Group 3 (Legal)": ["LEGAL_STATUTE", "CONTRACT_TYPE", "ORGANIZATION", "PERSON", "ROLE_TITLE", "GOV_AGENCY", "INTELLECTUAL_PROPERTY", "CLAUSE_TYPE", "JURISDICTION", "REGULATORY_FRAMEWORK"],
    "Group 4 (Medical)": ["DISEASE_SYNDROME", "MEDICATION", "BODY_PART", "MEDICAL_PROCEDURE", "BIOMARKER", "PATHOGEN", "DOSAGE", "MEDICAL_DEVICE", "HEALTHCARE_PROVIDER", "CLINICAL_TRIAL_ID"],
    "Group 5 (STEM)": ["CHEMICAL_COMPOUND", "SCIENTIFIC_UNIT", "PHYSICAL_CONSTANT", "EMAIL_ADDRESS", "PHONE_NUMBER", "ACADEMIC_JOURNAL", "EQUIPMENT_INSTRUMENT", "GENE_PROTEIN", "TAXON", "MATERIAL_TYPE"],
    "Group 6 (Geo-Spatial)": ["CITY", "COUNTRY", "REGION", "BUILDING", "TRANSIT_ROUTE", "VEHICLE_TYPE", "MANUFACTURER"],
    "Group 7 (Temporal/Event)": ["DATE", "TIME", "DURATION", "FREQUENCY", "HISTORICAL_EVENT", "URL", "NATURAL_DISASTER"]
}

ALL_VALID_LABELS = [label for groups in ONTOLOGY.values() for label in groups]

BATCHES = [
    {"theme": "Software & IT", "labels": ", ".join(ONTOLOGY["Group 1 (Software)"]), "model": "minimax/minimax-m2.7:nitro"},
    {"theme": "Financial & Market", "labels": ", ".join(ONTOLOGY["Group 2 (Financial)"]), "model": "deepseek/deepseek-v3.2"},
    {"theme": "Legal & Corporate", "labels": ", ".join(ONTOLOGY["Group 3 (Legal)"]), "model": "deepseek/deepseek-v3.2"},
    {"theme": "Biomedical & Healthcare", "labels": ", ".join(ONTOLOGY["Group 4 (Medical)"]), "model": "minimax/minimax-m2.7:nitro"},
    {"theme": "STEM & Academic", "labels": ", ".join(ONTOLOGY["Group 5 (STEM)"]), "model": "deepseek/deepseek-v3.2"},
    {"theme": "Geo-Spatial & Logistics", "labels": ", ".join(ONTOLOGY["Group 6 (Geo-Spatial)"]), "model": "deepseek/deepseek-v3.2"},
    {"theme": "Temporal & Event", "labels": ", ".join(ONTOLOGY["Group 7 (Temporal/Event)"]), "model": "deepseek/deepseek-v3.2"},
]

FORMATS = [
    "An excerpt from an SEC 10-K financial filing or quarterly earnings call transcript.",
    "A patient's clinical chart, doctor's notes, or a pharmacology trial summary.",
    "A formal legal brief, contract NDA, or regulatory compliance audit.",
    "An abstract from a peer-reviewed academic journal in physics or biology.",
    "A frantic Slack message between DevOps engineers debugging a server outage.",
    "A logistics and supply chain incident report regarding international shipping.",
    "A news article reporting on a recent historical event, disaster, or cultural festival."
]

SYSTEM_PROMPT = f"""
You are a strict Senior Machine Learning Data Engineer generating synthetic BIO-tagged training data for a 64-label Mixture of Experts (MoE) NER model.

GLOBAL ONTOLOGY (64 Labels):
{json.dumps(ONTOLOGY, indent=2)}

RULES:
1. Evaluate every word against ALL 64 labels. Never tag a known entity as "O".
2. Use strict whole-word BIO tagging (B-LABEL, I-LABEL, O). 
3. Treat ALL punctuation (commas, periods, quotes, parentheses, colons) as separate tokens tagged as "O". Do not attach punctuation to entity tokens.
4. Output ONLY valid JSON containing a single array named "dataset" of objects: {{"tokens": ["..."], "ner_tags": ["..."]}}.
5. CRITICAL: Output ONLY JSON. No Chain-of-Thought. No reasoning. No explanations. No markdown. Just the JSON object.
"""


MIN_VALID_THRESHOLD = 0.5
MAX_RETRIES = 3
WAL_FILE = "wal.json"

def validate_bio_logic(tokens, tags):
    if len(tokens) != len(tags):
        return False, f"Length mismatch: {len(tokens)} tokens vs {len(tags)} tags."
    
    for tag in tags:
        if tag == "O": continue
        if not (tag.startswith("B-") or tag.startswith("I-")):
            return False, f"Malformed BIO tag: {tag}"
            
        label_content = tag[2:]
        if label_content not in ALL_VALID_LABELS:
            return False, f"Hallucinated label not in 64-schema: {label_content}"
            
    return True, "Valid"

def count_existing_sentences(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def verify_sentence_quality(item):
    if not isinstance(item, dict):
        return False, "Item is not a dict"
    tokens = item.get("tokens", [])
    tags = item.get("ner_tags", [])
    if not tokens or not tags:
        return False, "Missing tokens or ner_tags"
    if len(tokens) < 5:
        return False, f"Sentence too short: {len(tokens)} tokens"
    return validate_bio_logic(tokens, tags)


def build_validation_report(batch, sample_limit=3):
    valid_batch = []
    reason_counts = Counter()
    invalid_examples = []
    for idx, item in enumerate(batch):
        is_valid, reason = verify_sentence_quality(item)
        if is_valid:
            valid_batch.append(item)
            continue

        reason_counts[reason] += 1
        if len(invalid_examples) < sample_limit:
            tokens = item.get("tokens") if isinstance(item, dict) else None
            tags = item.get("ner_tags") if isinstance(item, dict) else None
            invalid_examples.append(
                {
                    "index": idx,
                    "reason": reason,
                    "tokens_preview": _preview_sequence(tokens),
                    "tags_preview": _preview_sequence(tags),
                }
            )
    return ValidationReport(
        total=len(batch),
        valid_batch=valid_batch,
        reason_counts=dict(reason_counts),
        invalid_examples=invalid_examples,
    )


def log_validation_report(task_id, attempt, report):
    logger.info(
        "[Task %s] Attempt %s validation summary: %s/%s valid, %s invalid",
        task_id,
        attempt,
        report.valid_count,
        report.total,
        report.invalid_count,
    )
    if report.reason_counts:
        for reason, count in sorted(report.reason_counts.items(), key=lambda item: (-item[1], item[0])):
            logger.warning("[Task %s] validation failure reason: count=%s reason=%s", task_id, count, reason)
    for sample_no, sample in enumerate(report.invalid_examples, start=1):
        logger.warning(
            "[Task %s] invalid sample #%s index=%s reason=%s tokens=%s tags=%s",
            task_id,
            sample_no,
            sample["index"],
            sample["reason"],
            sample["tokens_preview"],
            sample["tags_preview"],
        )

def load_wal():
    if not os.path.exists(WAL_FILE):
        logger.info("[wal] no existing WAL at %s; starting fresh", os.path.abspath(WAL_FILE))
        return {"completed_tasks": [], "failed_tasks": [], "sentence_count": 0}
    with open(WAL_FILE, "r") as f:
        try:
            wal = json.load(f)
            logger.info(
                "[wal] loaded %s completed=%s failed=%s sentence_count=%s",
                os.path.abspath(WAL_FILE),
                len(wal.get("completed_tasks", [])),
                len(wal.get("failed_tasks", [])),
                wal.get("sentence_count", 0),
            )
            return wal
        except json.JSONDecodeError:
            logger.warning("[wal] corrupt WAL at %s; starting with empty state", os.path.abspath(WAL_FILE))
            return {"completed_tasks": [], "failed_tasks": [], "sentence_count": 0}

def save_wal(wal_state):
    temp_file = f"{WAL_FILE}.tmp"
    with open(temp_file, "w") as f:
        json.dump(wal_state, f, indent=2)
    os.replace(temp_file, WAL_FILE)
    logger.info(
        "[wal] saved completed=%s failed=%s sentence_count=%s path=%s",
        len(wal_state.get("completed_tasks", [])),
        len(wal_state.get("failed_tasks", [])),
        wal_state.get("sentence_count", 0),
        os.path.abspath(WAL_FILE),
    )

def mark_task_completed(wal_state, task_id, sentence_count):
    if task_id not in wal_state["completed_tasks"]:
        wal_state["completed_tasks"].append(task_id)
    wal_state["sentence_count"] = sentence_count
    logger.info("[Task %s] marking complete at sentence_count=%s", task_id, sentence_count)
    save_wal(wal_state)

def mark_task_failed(wal_state, task_id, error):
    if task_id not in wal_state["failed_tasks"]:
        wal_state["failed_tasks"].append({"task_id": task_id, "error": str(error)})
    logger.error("[Task %s] marking failed: %s", task_id, error)
    save_wal(wal_state)


class GenerationError(Exception):
    pass

async def generate_batch_with_retry(client, batch_config, task_id, semaphore, file_lock, timeout, output_file, wal_state, num_sentences=20):
    
    backoff = 2
    narrative = random.choice(FORMATS)
    target_model = batch_config["model"]
    
    prompt = f"""
    Task: Generate {num_sentences} sentences.
    Focus Theme: {batch_config['theme']}
    Primary Labels to feature heavily: {batch_config['labels']}
    Narrative Style: {narrative}
    
    Constraints: 
    - Sentences must be highly variable (15 to 40 words).
    - CROSS-POLLINATE: You MUST include at least 2 entities from completely unrelated groups in every sentence to teach the model context switching.
    - Ensure 'Hard Negatives' (pack multiple different entity types closely together).
    """
    logger.info(
        "[Task %s] prepared theme=%s model=%s requested_sentences=%s narrative=%s labels=%s",
        task_id,
        batch_config["theme"],
        target_model,
        num_sentences,
        narrative,
        batch_config["labels"],
    )

    async with semaphore:
        last_error = None
        for attempt in range(MAX_RETRIES):
            start_time = time.time()
            attempt_no = attempt + 1
            try:
                logger.info(
                    "[Task %s] Attempt %s/%s request start model=%s timeout=%ss output=%s",
                    task_id,
                    attempt_no,
                    MAX_RETRIES,
                    target_model,
                    timeout,
                    output_file,
                )
                response = await client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={ "type": "json_object" }, 
                    temperature=0.7,
                    timeout=timeout
                )

                choice = response.choices[0]
                content = choice.message.content or ""
                usage = getattr(response, "usage", None)
                logger.info(
                    "[Task %s] Attempt %s response received id=%s response_model=%s finish_reason=%s content_chars=%s usage=%s",
                    task_id,
                    attempt_no,
                    getattr(response, "id", None),
                    getattr(response, "model", None),
                    getattr(choice, "finish_reason", None),
                    len(content),
                    usage,
                )
                if os.environ.get("TRAINING_LOG_RAW_RESPONSE") == "1":
                    logger.info("[Task %s] Attempt %s raw response preview=%s", task_id, attempt_no, _preview_text(content, limit=2000))

                try:
                    raw_data = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(
                        "[Task %s] Attempt %s JSON parse failed line=%s col=%s msg=%s preview=%s",
                        task_id,
                        attempt_no,
                        e.lineno,
                        e.colno,
                        e.msg,
                        _preview_text(content),
                    )
                    raise

                logger.info(
                    "[Task %s] Attempt %s parsed JSON type=%s keys=%s",
                    task_id,
                    attempt_no,
                    type(raw_data).__name__,
                    list(raw_data.keys()) if isinstance(raw_data, dict) else "<not dict>",
                )
                batch = raw_data.get("dataset", [])
                logger.info(
                    "[Task %s] Attempt %s dataset extracted type=%s count=%s",
                    task_id,
                    attempt_no,
                    type(batch).__name__,
                    len(batch) if isinstance(batch, list) else "<not list>",
                )
                
                if not batch:
                    raise ValueError("API returned empty dataset array.")
                
                report = build_validation_report(batch)
                log_validation_report(task_id, attempt_no, report)
                
                valid_batch = report.valid_batch
                valid_ratio = len(valid_batch) / num_sentences
                if valid_ratio < MIN_VALID_THRESHOLD:
                    raise ValueError(f"Validation threshold not met: {len(valid_batch)}/{num_sentences} valid ({valid_ratio:.0%}), {report.invalid_count} invalid")

                elapsed = time.time() - start_time
                logger.info(f"[Task {task_id}] ✅ {len(valid_batch)}/{num_sentences} valid ({valid_ratio:.0%}) in {elapsed:.1f}s via {target_model}")
                
                async with file_lock:
                    logger.info("[Task %s] writing %s valid records to %s", task_id, len(valid_batch), output_file)
                    with open(output_file, "a", encoding="utf-8") as f:
                        for sentence_obj in valid_batch:
                            f.write(json.dumps(sentence_obj) + "\n")
                    
                    current_count = count_existing_sentences(output_file)
                    mark_task_completed(wal_state, task_id, current_count)
                        
                return valid_batch

            except Exception as e:
                last_error = e
                elapsed = time.time() - start_time
                logger.warning(f"[Task {task_id}] Attempt {attempt_no}/{MAX_RETRIES} failed in {elapsed:.1f}s: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    sleep_time = backoff * (2 ** attempt) + random.uniform(0, 1)
                    logger.info("[Task %s] sleeping %.1fs before retry", task_id, sleep_time)
                    await asyncio.sleep(sleep_time)
        
        async with file_lock:
            mark_task_failed(wal_state, task_id, str(last_error))
        
        raise GenerationError(f"Task {task_id} failed after {MAX_RETRIES} retries: {last_error}")


async def main():
    try:
        or_setup = load_openrouter_config()
    except Exception as e:
        logger.error(f"Configuration Error: {e}")
        return

    client = AsyncOpenAI(
        base_url=or_setup["base_url"],
        api_key=or_setup["api_key"]
    )
    
    settings = get_generation_settings()
    global WAL_FILE
    WAL_FILE = settings.wal_file
    logger.info(
        "[settings] total_sentences_target=%s sentences_per_call=%s total_calls=%s max_concurrent_calls=%s output_file=%s wal_file=%s",
        settings.total_sentences_target,
        settings.sentences_per_call,
        settings.total_calls,
        settings.max_concurrent_calls,
        settings.output_file,
        settings.wal_file,
    )
    
    wal_state = load_wal()
    existing_sentences = count_existing_sentences(settings.output_file)
    completed_task_ids = set(wal_state.get("completed_tasks", []))
    failed_tasks = wal_state.get("failed_tasks", [])
    
    if failed_tasks:
        logger.error(f"❌ Found {len(failed_tasks)} permanently failed tasks in WAL:")
        for ft in failed_tasks:
            logger.error(f"   Task {ft['task_id']}: {ft['error']}")
        logger.error("Fix the issue and delete wal.json to retry, or manually investigate.")
        return
    
    if existing_sentences >= settings.total_sentences_target:
        logger.info(f"✅ Dataset complete: {existing_sentences}/{settings.total_sentences_target} sentences")
        return

    pending_tasks = [i for i in range(settings.total_calls) if (i + 1) not in completed_task_ids]
    
    logger.info(f"Target: {settings.total_sentences_target} sentences")
    logger.info(f"Completed: {len(completed_task_ids)} tasks ({existing_sentences} sentences)")
    logger.info(f"Pending: {len(pending_tasks)} tasks")
    logger.info(f"Model routing: MiniMax M2.7 (Software/Medical) | DeepSeek V3.2 (Others)")

    if not pending_tasks:
        logger.info("No pending tasks.")
        return

    semaphore = asyncio.Semaphore(settings.max_concurrent_calls)
    file_lock = asyncio.Lock()

    tasks = []
    for i in pending_tasks:
        current_batch = BATCHES[i % len(BATCHES)]
        task_id = i + 1
        
        tasks.append(
            generate_batch_with_retry(
                client, 
                current_batch, 
                task_id, 
                semaphore, 
                file_lock, 
                or_setup["timeout"],
                settings.output_file,
                wal_state,
                num_sentences=settings.sentences_per_call,
            )
        )

    logger.info(f"Starting {len(tasks)} tasks (max {settings.max_concurrent_calls} concurrent)...")
    
    try:
        await asyncio.gather(*tasks)
    except GenerationError as e:
        logger.error(f"\n❌ FATAL: {e}")
        logger.error("Script terminated. Check wal.json for details. Fix the issue and re-run.")
        return

    final_count = count_existing_sentences(settings.output_file)
    logger.info(f"\n✅ COMPLETE: {final_count}/{settings.total_sentences_target} sentences in {settings.output_file}")

if __name__ == "__main__":
    asyncio.run(main())

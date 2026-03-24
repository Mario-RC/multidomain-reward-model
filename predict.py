from typing import Dict, List
from datetime import datetime
import torch
from argparse import ArgumentParser
from transformers import AutoTokenizer
from modeling_custom import RewardModelWithGating
from config_utils import load_yaml_config
from utils import _resolve_inference_model_path

class MultiDomainRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=None, truncation=True, max_length=4096):
        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # `device_map="auto"` is not supported by this custom model class because
        # Transformers requires `_no_split_modules` for automatic sharding.
        if device_map == "auto":
            device_map = {"": 0} if torch.cuda.is_available() else None

        self.model = RewardModelWithGating.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = next(self.model.parameters()).device
        self.max_length = max_length
        self.model.eval()

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        encoding = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        )
        if isinstance(encoding, torch.Tensor):
            input_ids = encoding.to(self.device)
            attention_mask = None
        else:
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding.get("attention_mask")
            attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            score = output.score.float().item()
        return {"score": score}


def main() -> None:
    print(f"\n### Predict started at {datetime.now().isoformat()} ###")
    parser = ArgumentParser(description="Run quick prediction comparison using packaged reward model.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--model_path", type=str, default=None, help="Optional override for packaged model path.")
    parser.add_argument("--model_parent_dir", type=str, default="model", help="Optional packaged model parent directory.")
    parser.add_argument("--model_name", type=str, default=None, help="Optional packaged model directory name.")
    args = parser.parse_args()

    config = load_yaml_config(args.config_path)
    model_path = _resolve_inference_model_path(config, args.model_path, args.model_parent_dir, args.model_name)
    print(f"Loading model: {model_path}")

    # Initialize the local model.
    rm = MultiDomainRMPipeline(model_path)

    examples = [
        # ── 1. Empathy: grief & loss (multi-turn) ──
        {
            "label": "Empathy – Grief (warm vs dismissive)",
            "good": [
                {"role": "user", "content": "My old cat is very sick and we have to put her down next week."},
                {"role": "assistant", "content": "I'm so sorry to hear that. Losing a companion you've loved for years is incredibly painful. How are you holding up?"},
                {"role": "user", "content": "She's been with me for 18 years. I can't imagine life without her."},
                {"role": "assistant", "content": "Eighteen years — she's been there through so much of your life. It makes sense that you're dreading this. Take all the time you need to grieve; that bond deserves to be honoured."},
            ],
            "bad": [
                {"role": "user", "content": "My old cat is very sick and we have to put her down next week."},
                {"role": "assistant", "content": "Well, 18 is a long life for a cat. At least you got your money's worth."},
                {"role": "user", "content": "She's been with me for 18 years. I can't imagine life without her."},
                {"role": "assistant", "content": "You could always get another cat. Shelters are full of them."},
            ],
        },
        # ── 2. Empathy: professional setback ──
        {
            "label": "Empathy – Job rejection (validating vs minimising)",
            "good": [
                {"role": "user", "content": "I didn't get the promotion I've been working toward for two years."},
                {"role": "assistant", "content": "That must be really disheartening after putting in so much effort. It's okay to feel frustrated — two years of dedication is significant, and your disappointment is completely valid."},
            ],
            "bad": [
                {"role": "user", "content": "I didn't get the promotion I've been working toward for two years."},
                {"role": "assistant", "content": "Promotions are competitive. You should update your resume and apply elsewhere if you're unhappy."},
            ],
        },
        # ── 3. Coherence: topic drift ──
        {
            "label": "Coherence – Staying on topic vs drifting",
            "good": [
                {"role": "user", "content": "Can you recommend a good Italian restaurant nearby?"},
                {"role": "assistant", "content": "Sure! Trattoria Bella on 5th Street has amazing homemade pasta and a great wine list. They also do a wood-fired pizza that's very popular."},
                {"role": "user", "content": "Do they have outdoor seating?"},
                {"role": "assistant", "content": "Yes, they have a lovely patio out back. It's heated in the winter too, so you can sit outside year-round."},
            ],
            "bad": [
                {"role": "user", "content": "Can you recommend a good Italian restaurant nearby?"},
                {"role": "assistant", "content": "Italy is a beautiful country. The Roman Empire was one of the most influential civilisations in history."},
                {"role": "user", "content": "Do they have outdoor seating?"},
                {"role": "assistant", "content": "Outdoor activities are great for mental health. You should try hiking sometime."},
            ],
        },
        # ── 4. Coherence: logical consistency ──
        {
            "label": "Coherence – Logical consistency vs contradiction",
            "good": [
                {"role": "user", "content": "Is the store open on Sundays?"},
                {"role": "assistant", "content": "Yes, we're open on Sundays from 10 AM to 6 PM."},
                {"role": "user", "content": "Great, I'll come by around noon then."},
                {"role": "assistant", "content": "Perfect, noon works. We tend to be less busy around that time so you'll get good service."},
            ],
            "bad": [
                {"role": "user", "content": "Is the store open on Sundays?"},
                {"role": "assistant", "content": "Yes, we're open on Sundays from 10 AM to 6 PM."},
                {"role": "user", "content": "Great, I'll come by around noon then."},
                {"role": "assistant", "content": "Sorry, we're closed on Sundays. You'll have to come on a weekday."},
            ],
        },
        # ── 5. Commonsense: cause and effect ──
        {
            "label": "Commonsense – Cause/effect reasoning",
            "good": [
                {"role": "user", "content": "I left my ice cream on the dashboard while I went into the store."},
                {"role": "assistant", "content": "Oh no, it's probably melted by now, especially if the car was in the sun. You might want to check it quickly."},
            ],
            "bad": [
                {"role": "user", "content": "I left my ice cream on the dashboard while I went into the store."},
                {"role": "assistant", "content": "That's a great spot for it. The dashboard keeps things cool because it's near the windshield."},
            ],
        },
        # ── 6. Commonsense: social norms & reactions ──
        {
            "label": "Commonsense – Social norms (wedding)",
            "good": [
                {"role": "user", "content": "My best friend just told me she's getting married next month!"},
                {"role": "assistant", "content": "That's wonderful news! You must be so happy for her. Are you going to be part of the wedding party?"},
            ],
            "bad": [
                {"role": "user", "content": "My best friend just told me she's getting married next month!"},
                {"role": "assistant", "content": "Marriage is a legal contract. She should make sure to get a prenuptial agreement first."},
            ],
        },
        # ── 7. Multicultural: respecting cultural practices ──
        {
            "label": "Multicultural – Respecting Ramadan fasting",
            "good": [
                {"role": "user", "content": "I'm fasting for Ramadan this month, so I won't be joining for lunch."},
                {"role": "assistant", "content": "Thanks for letting me know! I hope the fasting goes well. If you'd like, we could schedule our meetings around Iftar time so it's more comfortable for you."},
            ],
            "bad": [
                {"role": "user", "content": "I'm fasting for Ramadan this month, so I won't be joining for lunch."},
                {"role": "assistant", "content": "Skipping meals isn't healthy. You should eat something — it's not good to go without food all day."},
            ],
        },
        # ── 8. Multicultural: cross-cultural communication ──
        {
            "label": "Multicultural – Navigating cultural differences (bowing vs handshake)",
            "good": [
                {"role": "user", "content": "I'm meeting my Japanese business partners tomorrow. Any advice on greetings?"},
                {"role": "assistant", "content": "A slight bow is the traditional greeting in Japanese business culture. Offer your business card with both hands and receive theirs the same way — it shows respect. They may also offer a handshake since they're meeting a foreigner, so follow their lead."},
            ],
            "bad": [
                {"role": "user", "content": "I'm meeting my Japanese business partners tomorrow. Any advice on greetings?"},
                {"role": "assistant", "content": "Just give them a firm handshake and look them in the eye. That's how you show confidence in any business setting worldwide."},
            ],
        },
        # ── 9. Empathy + Coherence: anxiety multi-turn ──
        {
            "label": "Empathy + Coherence – Anxiety (supportive & coherent vs cold & drifting)",
            "good": [
                {"role": "user", "content": "I've been having panic attacks before every presentation at work."},
                {"role": "assistant", "content": "That sounds really exhausting and scary. Panic attacks can make you feel completely out of control. Have you been able to talk to anyone about what you're going through?"},
                {"role": "user", "content": "Not really. I'm afraid my boss will think I'm not capable."},
                {"role": "assistant", "content": "That fear makes a lot of sense — there's so much pressure to seem strong at work. But struggling with anxiety doesn't mean you're incapable. A lot of people deal with this. Would you feel comfortable talking to a therapist? It can stay completely separate from work."},
            ],
            "bad": [
                {"role": "user", "content": "I've been having panic attacks before every presentation at work."},
                {"role": "assistant", "content": "Presentations are part of most jobs. You just need more practice."},
                {"role": "user", "content": "Not really. I'm afraid my boss will think I'm not capable."},
                {"role": "assistant", "content": "Speaking of bosses, have you considered switching to a remote role? The tech industry has a lot of openings right now."},
            ],
        },
        # ── 10. Commonsense + Multicultural: travel advice ──
        {
            "label": "Commonsense + Multicultural – Travel advice (India)",
            "good": [
                {"role": "user", "content": "I'm visiting a Hindu temple in India next week. What should I know?"},
                {"role": "assistant", "content": "You'll want to remove your shoes before entering — there's usually a rack or attendant near the entrance. Dress modestly with shoulders and knees covered. If food is offered as prasad, it's polite to accept it with your right hand. Also avoid pointing your feet toward the shrine, as it's considered disrespectful."},
            ],
            "bad": [
                {"role": "user", "content": "I'm visiting a Hindu temple in India next week. What should I know?"},
                {"role": "assistant", "content": "Temples are tourist attractions, so just wear whatever is comfortable and take plenty of photos inside. You'll have a great time."},
            ],
        },
    ]

    print(f"{'='*70}")
    print(f"{'Example':<55} {'Good':>7} {'Bad':>7}")
    print(f"{'='*70}")
    for ex in examples:
        score_good = rm(ex["good"])
        score_bad = rm(ex["bad"])
        print(f"{ex['label']:<55} {score_good['score']:>7.4f} {score_bad['score']:>7.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
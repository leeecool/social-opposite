# social-opposite
import re
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ---------------------------
# 모델 로딩 (한국어 지원)
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("jhgan/ko-sroberta-multitask")

@st.cache_resource
def load_candidates():
    return {
        "찬성": [
            "해당 정책은 사회적 약자에게 실질적인 지원을 제공하여 생계 안정에 기여합니다.",
            "장기적으로 불평등 완화와 사회 통합에 도움이 될 수 있습니다.",
            "초기 비용은 크지만, 사회적 편익(범죄 감소, 건강 개선 등)이 더 클 수 있습니다."
        ],
        "반대": [
            "해당 정책은 예산 부담이 커서 지속 가능성이 낮을 수 있습니다.",
            "대상자 선정과 형평성 문제로 사회적 갈등이 발생할 수 있습니다.",
            "정책이 남용되거나 의도치 않은 부작용을 초래할 가능성이 있습니다."
        ],
        "중립": [
            "입력이 명확한 찬반을 담고 있지 않아 중립적인 관점으로 보입니다."
        ]
    }

@st.cache_resource
def embed_candidates(model, candidates):
    embs = {}
    for k, lst in candidates.items():
        embs[k] = model.encode(lst, convert_to_tensor=True)
    return embs

def classify_polarity(text):
    text_proc = text.lower()
    pos = ["좋다","필요","찬성","지지","도움","필수"]
    neg = ["반대","문제","우려","불필요","해롭다","부정"]
    def cnt(keys):
        c=0
        for k in keys:
            pattern = rf"(?<![가-힣A-Za-z0-9]){re.escape(k)}(?![가-힣A-Za-z0-9])"
            c += len(re.findall(pattern, text_proc))
        return c
    p = cnt(pos); n = cnt(neg)
    if p==0 and n==0: return "중립", 0.0
    if p>n: return "찬성", round((p-n)/(p+n),2)
    if n>p: return "반대", round((n-p)/(p+n),2)
    return "중립", 0.0

def get_opposite_label(l):
    return "반대" if l=="찬성" else ("찬성" if l=="반대" else "중립")

def generate_counter(model, emb_cache, candidates, text, stance, top_k=3):
    opp = get_opposite_label(stance)
    if opp=="중립": return "입력이 중립적입니다. 찬성/반대가 분명한 문장을 입력해 주세요."
    text_emb = model.encode(text, convert_to_tensor=True)
    c_embs = emb_cache[opp]
    c_texts = candidates[opp]
    sims = util.cos_sim(text_emb, c_embs)[0].cpu().tolist()
    idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
    out = [f"{i+1}. (유사도:{round(sims[idx],3)}) {c_texts[idx]}" for i,idx in enumerate(idxs)]
    return f"입력 관점: {stance}\n\n" + "\n".join(out)

st.set_page_config(page_title="사회 정보 반대 관점 추천", page_icon="📰")
st.title("사회 정보 반대 관점 추천 앱")
st.write("사회 분야 콘텐츠를 입력하면, 자동으로 **반대되는 관점**의 정보를 추천합니다. (연구/실험용)")

user_text = st.text_area("사회 관련 글을 입력하세요.", placeholder="예: 나는 청년 기본소득이 필요하다고 생각한다.")
if st.button("생성"):
    if not user_text.strip():
        st.warning("내용을 입력하세요.")
    else:
        model = load_model()
        candidates = load_candidates()
        emb_cache = embed_candidates(model, candidates)
        stance, conf = classify_polarity(user_text)
        st.markdown(f"### 🧭 추정된 성향: **{stance}** (신뢰도: {conf})")
        st.write("---")
        st.write(generate_counter(model, emb_cache, candidates, user_text, stance))

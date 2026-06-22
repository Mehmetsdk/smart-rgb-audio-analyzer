import os, json, colorsys, math
import numpy as np
import librosa
import anthropic

AUDIO_PATH  = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\theless.wav"
OUTPUT_JSON = r"C:\Users\Mehmet Sadık Gürler\Desktop\BM-Proje\light_script.json"
WINDOW_SEC  = 1.0
BATCH_SEC   = 10.0     # 10 window/batch — Claude tüm listeyi güvenilir tamamlar
SIGMA       = 3.0
MODEL       = "claude-haiku-4-5"

# ── Şarkı bilgisi — başka şarkı için sadece burayı değiştir ──────────────────
SONG_TITLE  = "The Less I Know The Better"
SONG_ARTIST = "Tame Impala"
SONG_INFO   = "Bittersweet psychedelic pop (2015). Emotional arc: jealous longing → euphoric release → dreamy nostalgia → wistful acceptance."
# ─────────────────────────────────────────────────────────────────────────────

CONTEXT = f"""Song: "{SONG_TITLE}" by {SONG_ARTIST}.
{SONG_INFO}

You control 3 RGB light zones. Analyze what a human listener feels at each moment
and assign an independent color to each zone:
  zone1 = outer ring  — the BEAT / RHYTHM / physical feeling
  zone2 = middle ring — the MELODY / HARMONY / emotional mood
  zone3 = inner disc  — the VOCAL / LYRICAL / intimate feeling

Each zone picks its own color freely — they do not have to match or be related.
A driving beat can be crimson while the melody floats in teal and the voice glows soft gold."""


def extract(path):
    print("Ses yükleniyor...")
    y, sr = librosa.load(path, sr=None, mono=True)
    ws = int(WINDOW_SEC * sr)
    n  = len(y) // ws
    print(f"{n} kare çıkarılıyor...")
    frames = []
    for i in range(n):
        seg = y[i*ws:(i+1)*ws]
        rms     = float(np.mean(librosa.feature.rms(y=seg)))
        onset   = float(np.mean(librosa.onset.onset_strength(y=seg, sr=sr)))
        centroid= float(np.mean(librosa.feature.spectral_centroid(y=seg, sr=sr)))
        chroma  = librosa.feature.chroma_stft(y=seg, sr=sr)
        note    = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][
                      int(np.argmax(np.mean(chroma, axis=1)))]
        frames.append({"t": round(i*WINDOW_SEC, 1),
                        "rms": round(rms,4), "onset": round(onset,2),
                        "centroid": round(centroid/(sr/2),3), "note": note})
    return frames


def ask_claude(client, batch):
    lines = "\n".join(
        f"t={f['t']}s  rms={f['rms']}  onset={f['onset']}  bright={f['centroid']}  note={f['note']}"
        for f in batch)
    prompt = f"""{CONTEXT}

Audio features ({len(batch)} windows, 1s each):
{lines}

Return ONLY a JSON array — one object per window, same order:
[{{"t":0.0,"emotion":"...","zone1":[r,g,b],"zone2":[r,g,b],"zone3":[r,g,b]}},...]"""

    resp = client.messages.create(
        model=MODEL, max_tokens=4096,
        messages=[{"role":"user","content":prompt}])
    text = resp.content[0].text.strip()
    s, e = text.find("["), text.rfind("]")+1
    if s == -1:
        raise ValueError(text[:300])
    chunk = text[s:e]
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        # Liste yarıda kesilmişse: tamamlanmış objeleri tek tek kurtar
        objs = []
        depth = 0; start = None
        for i, ch in enumerate(chunk):
            if ch == "{":
                if depth == 0: start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        objs.append(json.loads(chunk[start:i+1]))
                    except json.JSONDecodeError:
                        pass
        if not objs:
            raise
        return objs


def smooth(colors, sigma):
    n = len(colors)
    r = max(1, int(math.ceil(3*sigma)))
    k = [math.exp(-0.5*((i-r)/sigma)**2) for i in range(2*r+1)]
    ks = sum(k); k = [w/ks for w in k]
    def to_hls(c): return colorsys.rgb_to_hls(c[0]/255, c[1]/255, c[2]/255)
    def fr_hls(h,l,s):
        r2,g2,b2 = colorsys.hls_to_rgb(h%1,l,s)
        return [int(r2*255), int(g2*255), int(b2*255)]
    hls = [to_hls(c) for c in colors]
    out = []
    for i in range(n):
        rh=hls[i][0]; wh=wl=ws=tw=0.0
        for ki,kw in enumerate(k):
            j=i-r+ki
            if j<0 or j>=n: continue
            h,l,s=hls[j]; dh=h-rh
            if dh>.5: dh-=1
            if dh<-.5: dh+=1
            wh+=(rh+dh)*kw; wl+=l*kw; ws+=s*kw; tw+=kw
        out.append(fr_hls(wh/tw, wl/tw, ws/tw) if tw else colors[i])
    return out


def main():
    key = os.environ.get("ANTHROPIC_API_KEY","")
    if not key:
        print("ANTHROPIC_API_KEY eksik"); return

    client = anthropic.Anthropic(api_key=key)
    frames = extract(AUDIO_PATH)
    bsz    = int(BATCH_SEC/WINDOW_SEC)
    nb     = math.ceil(len(frames)/bsz)
    print(f"{nb} batch Claude'a gönderiliyor...\n")

    # Eksik kareler None kalır → sonradan komşulardan doldurulur
    N = len(frames)
    z1r = [None]*N; z2r = [None]*N; z3r = [None]*N; ems = [None]*N
    cl  = lambda v: max(0, min(255, int(v)))

    def fill_batch(batch, lo, results):
        rmap = {round(float(r["t"]),1): r for r in results if isinstance(r,dict)}
        ok = 0
        for k, f in enumerate(batch):
            ts  = round(f["t"],1)
            res = rmap.get(ts)
            if res is None and rmap:
                closest = min(rmap, key=lambda kk: abs(kk-ts))
                if abs(closest-ts) < 1.2: res = rmap[closest]
            if res and "zone1" in res and "zone2" in res and "zone3" in res:
                idx = lo + k
                z1r[idx] = [cl(res["zone1"][i]) for i in range(3)]
                z2r[idx] = [cl(res["zone2"][i]) for i in range(3)]
                z3r[idx] = [cl(res["zone3"][i]) for i in range(3)]
                ems[idx] = str(res.get("emotion",""))
                ok += 1
        return ok

    for bi in range(nb):
        lo,hi = bi*bsz, min((bi+1)*bsz, len(frames))
        batch = frames[lo:hi]
        print(f"  Batch {bi+1}/{nb}  t={batch[0]['t']}s–{batch[-1]['t']}s  ",
              end="", flush=True)

        ok = 0
        for attempt in range(2):   # 1 deneme + 1 retry
            try:
                results = ask_claude(client, batch)
                ok = fill_batch(batch, lo, results)
                if ok >= len(batch) * 0.8:   # %80+ tamamsa yeterli
                    break
                print(f"(eksik:{len(batch)-ok}, retry) ", end="", flush=True)
            except Exception as e:
                print(f"(hata, retry: {str(e)[:40]}) ", end="", flush=True)

        first = ems[lo] if ems[lo] else "(komşudan)"
        print(f"→ {ok}/{len(batch)} ok  {first}")

    # ── Eksik kareleri komşulardan doldur (forward+backward fill) ────────────
    def fill_gaps(arr, default):
        # Önce ileri doldur
        last = None
        for i in range(len(arr)):
            if arr[i] is not None: last = arr[i]
            elif last is not None: arr[i] = last
        # Sonra geri doldur (baştaki boşluklar için)
        nxt = None
        for i in range(len(arr)-1, -1, -1):
            if arr[i] is not None: nxt = arr[i]
            elif nxt is not None: arr[i] = nxt
        # Hâlâ None varsa default
        for i in range(len(arr)):
            if arr[i] is None: arr[i] = default
        return arr

    missing = sum(1 for x in z1r if x is None)
    if missing:
        print(f"{missing} eksik kare komşulardan dolduruluyor…")
    fill_gaps(z1r, [120,40,160]); fill_gaps(z2r, [70,40,170]); fill_gaps(z3r, [40,110,150])
    fill_gaps(ems, "")

    print("Smoothing...")
    s1,s2,s3 = smooth(z1r,SIGMA), smooth(z2r,SIGMA), smooth(z3r,SIGMA)

    data = [{"timestamp": frames[i]["t"], "emotion": ems[i],
             "artist": SONG_ARTIST, "title": SONG_TITLE,
             "zone1": s1[i], "zone2": s2[i], "zone3": s3[i]}
            for i in range(len(frames))]

    with open(OUTPUT_JSON,"w",encoding="utf-8") as f:
        json.dump(data,f,indent=2,ensure_ascii=False)

    print(f"\nTamamlandı — {len(data)} kare → {OUTPUT_JSON}")
    print("Başlatmak için: python simulator.py")


if __name__ == "__main__":
    main()

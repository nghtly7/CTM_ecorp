# Algoritmo finale — DWT + DCT + Spread-Spectrum + Fourier–Mellin

---

## Sommario

Algoritmo ibrido per watermarking robusto e veloce, composto da:

* Embedding: DWT(2) → DCT(8×8) mid-freq spread-spectrum + ECC + pilot tones + repliche
* Detection: riallineamento Fourier–Mellin (log-polar + phase correlation) → estrazione DWT/DCT → majority + ECC → decisione
* Fallback gerarchico (brute-force non riallineato → ORB limitato) con timeouts per rispettare `<= 5s`.
* `state.pkl` leggero con template e parametri necessari.

---

## File / API richiesti (convenzioni)

* `embedding(input1, input2) -> output1`

  * `input1`: path immagine originale BMP (512×512 grayscale)
  * `input2`: path watermark binario (file con 1024 bit)
  * `output1`: salva immagine watermarked con nome `groupname imageName.bmp`
  * Non stampare nulla, non aprire GUI.
* `detection(input1, input2, input3) -> output1, output2`

  * `input1`: path immagine originale (per calcolo WPSNR)
  * `input2`: path immagine watermarked (usata come riferimento)
  * `input3`: path immagine attaccata (da verificare)
  * `output1`: 1 se watermark presente, 0 altrimenti
  * `output2`: WPSNR(watermarked, attacked) (float)

---

## Struttura `state.pkl` (salvare al termine di embedding)

```text
{
  "LP_ref": np.float32 array (360,200),   # log-polar magnitude template
  "M_embed": np.bool array,               # mappa posizioni embedding (index map)
  "seed": int,                            # PRNG seed
  "ecc_params": dict,                     # {type, n, k, t}
  "alpha_matrix": np.float32 array (8,8), # α(u,v) HVS matrix per DCT coefficients
  "tau": float                            # soglia τ scelta via ROC
}
```

---

# Parametri raccomandati (base, da tarare)

* Wavelet: **biorthogonal 9/7** — livelli DWT = **2**
* DCT block: **8×8**
* Zigzag mid-freq positions: **10–30** (o indici 2–4 in u/v)
* Replica factor (coeff/bit): **R = 16 ÷ 32**
* Alpha base: **α0 = 0.008** ; alpha_matrix con β ≈ 0.6 (vedi HVS)
* ECC: **BCH** o **Reed-Solomon** con ridondanza ≈ **30%**
* LP_ref size: **360×200** (log-polar remap)
* Pilot tone amplitude: **0.6 (scaled)**
* τ (threshold) via ROC: target FPR 1–5% → iniziale τ ≈ **0.85–0.9**
* Detection time target: **< 5 s** (tipico 1.5–3 s)

---

# ALGORITMO — Dettaglio operativo

## 1) EMBEDDING (funzione `embedding(input1,input2)`)

### Step descrittivi

1. **Carica** `I_orig` (512×512 grayscale), converti a `float32` in [0,1].
2. **Pre-window**: applica finestra 2D Hanning per ridurre leakage FFT.
3. **Pilot tones**: inietti 4–6 piccoli segnali a bassa frequenza (bassa magnitudine) su posizioni predefinite nelle basse bande (usati per check rapido).
4. **DWT level 2**: calcola DWT (biorthogonal 9/7) → estrai LH2, HL2 (target bands).
5. **Per ogni subband target**:

   * Se necessario rimappa dimensione per essere multipla di 8.
   * Dividi in blocchi 8×8 e applica DCT2 (batch vectorizzato).
   * Costruisci `M_embed` (mappa posizioni dei coefficienti selezionati in ogni blocco).
6. **Watermark + ECC**:

   * Leggi i 1024 bit `W_bin`.
   * Applica ECC → ottieni `payload_bits` (N_payload).
   * PRNG = `RandomState(seed)` con `seed = SHA256(group_password)` (o generato casualmente e salvato).
   * Permuta `payload_bits` con PRNG e mappa ogni bit su `R` coefficienti secondo `M_embed`.
7. **Spread-spectrum insertion**:

   * Per ogni coeff selezionato: `c' = c + α(u,v) * s * bit`, dove:

     * `s` ∈ {+1, −1} pseudo-random per coeff (dalla stessa PRNG)
     * `α(u,v)` è tratto da `alpha_matrix` (8×8) scalata dalla luminanza locale (masking HVS).
8. **Repliche**: ripeti embedding su HL2 e LH2 (o su HL1/LH1 e HL2/LH2) per ridondanza.
9. **Inverse DCT / Inverse DWT** → ricostruisci immagine. Clip in [0,1] → converti uint8.
10. **Calcola WPSNR(I_orig, I_wm)** → se < 35 dB (fallimento) abbassa `α` o repliche; target ideale 54–66 dB (tarare).
11. **Calcola LP_ref**:

    * FFT magnitude di `I_wm`, `mag = log(1 + |FFT|)`, applica bandpass su bande medie, remap log-polar → `LP_ref`.
12. **Salva `state.pkl`** contenente `LP_ref`, `M_embed`, `seed`, `ecc_params`, `alpha_matrix`, `tau` (tau da ROC / test).
13. **Salva file `groupname imageName.bmp`** (output1).

### Note implementative

* Vectorizza DCT su array (reshape (nblocks,8,8) → dct2 per blocco).
* Calibrare `alpha_matrix` usando mappa di varianza locale: aree con maggior varianza → `α` maggiore.
* Non stampare o aprire finestre.

---

## 2) DETECTION (funzione `detection(input1,input2,input3)`)

### Obiettivi

* Riallineare `input3` a `input2` tramite Fourier–Mellin (principale), poi estrarre e confrontare watermark.
* Gerarchia fallback: (1) Fourier–Mellin, (2) estrazione non riallineata rapida, (3) ORB limitato (ultimo tentativo con timeout).

### Pipeline (procedura)

1. **Carica** `I_orig`, `I_wm` (input2), `I_att` (input3).
2. **Carica `state.pkl`** (LP_ref, M_embed, seed, ecc_params, alpha_matrix, tau).
3. **Riallineamento (Fourier–Mellin)**:

   * `mag_att = log(1 + |FFT(I_att * Hanning)|)`
   * Remap `LP_att = logpolar(mag_att, center)` (stesse dimensioni LP_ref).
   * `shift = phase_correlation(LP_ref, LP_att)` → ottieni `(dx, dy)` → stima `scale, rotation`.
   * Se peak_corr ≥ `corr_threshold` (es. 0.30) → applica inverse scaling/rotation su `I_att` → `I_aligned`.
   * ELSE → go to Step 4 (tentativo rapida senza riallineamento).
4. **Estrazione rapida (senza riallineamento)**:

   * Applica DWT(2) a `I_att`, DCT 8×8, recupera coefficienti secondo `M_embed`, calcola correlazioni col PRNG `P`.
   * Majority decode + ECC decode → se `sim ≥ tau_fast` (es. 0.7) → decisione positiva (return 1, WPSNR).
   * ELSE → go to Step 5.
5. **Fallback ORB (ultimo tentativo)**:

   * Start timer; timeout hard = 1.0 s (o ciò che rimane prima di 5 s).
   * Extract ORB keypoints limiting to **N=50** top responses.
   * Match `I_wm` ↔ `I_att`, RANSAC (max_iter 500) per stimare affin transform; se match ok → warp `I_att` → `I_aligned_orb`.
   * Se timeout o failed → **return** `output1 = 0, output2 = WPSNR(I_wm, I_att)` (no watermark).
6. **Estrarre watermark dall’immagine riallineata** (`I_aligned` o `I_aligned_orb`):

   * DWT(2) → DCT 8×8 → estrae coeff via `M_embed`.
   * Per ogni bit: correlazione con `P` → voto (+/-).
   * Majority vote tra repliche (LH2 vs HL2).
   * ECC decode → ottieni `W_extracted`.
7. **Calcola similarità**:

   * `sim = normalized_correlation(W_ref, W_extracted)` (dot / norms).
   * Se `sim >= tau` (tau da ROC) **AND** `wpsnr(I_wm, I_att) >= 35 dB` → `output1 = 1` (watermark presente).
   * Altrimenti `output1 = 0`.
8. **Output**:

   * `output1` (0/1), `output2` = WPSNR(I_wm, I_att).

### Timing e robustezza

* Fourier–Mellin + estrazione vectorizzata → tipico < 2 s su CPU moderne.
* Fallback ORB limitato aggiunge fino a ~1 s, ma è eseguito solo raramente.
* Timeout assoluto: non superare 5 s, preferibile restituire 0 piuttosto che superare il limite.

---

## 3) TESTING & ROC (procedura da eseguire offline)

1. **Dataset**: 30–50 immagini naturali 512×512. Per ogni immagine: genera `I_wm`.
2. **Attacchi** (range):

   * AWGN σ ∈ {2,4,8,12}
   * JPEG QF ∈ {30,50,70,90}
   * Median k ∈ {3,5,7}
   * Resize scales ∈ {0.5, 0.75, 0.9, 1.1, 1.5} (resize+reinterp back to 512)
   * Blur σ ∈ {0.5,1,2}
   * Sharpen: weak→strong kernels
3. **Per ogni attacco**: estrae `sim` e label (1 = watermark presente, 0 = random watermark).
4. **Genera ROC**, scegli `tau` nel range che produce **FPR 1–5%** (tradeoff robustezza vs falsi positivi).
5. Imposta `tau` e salva in `state.pkl`.

---

## 4) Suggerimenti pratici di implementazione

* **Vectorizzazione**: evitare loop Python su pixel/blocchi; usare reshape e funzioni DCT batch (`scipy.fftpack.dct` o `cv2.dct` su patches).
* **FFT**: usare `numpy.fft` o `pyfftw` per velocità; applica Hanning window 2D prima di FFT.
* **Log-polar**: usa `cv2.logPolar` (OpenCV) o implementazione equivalente con anti-aliasing.
* **ECC**: usare libreria consolidata (e.g. `pyfinite` o `reedsolo`); testare robustezza e velocità.
* **Profiling**: misurare tempi con `time.perf_counter()` e tracciare percentili sul dataset.
* **Logging**: salva seed, alpha_matrix, parametri ECC e ROC τ in `state.pkl` (no immagini).
* **WPSNR**: implementa la funzione con la stessa definizione usata in laboratorio (usare WPSNR fornita).

---

## 5) Pseudocodice essenziale

### Embedding (schematic)

```python
def embedding(input1, input2):
    I = load_image(input1)                 # float32 0..1
    W = load_watermark_bits(input2)        # 1024 bits
    # pre-window
    I_win = apply_hanning(I)
    # DWT level2
    coeffs = dwt2(I_win, wavelet='bior9.7', level=2)
    LH2, HL2 = coeffs['LH2'], coeffs['HL2']
    # DCT 8x8 on LH2 and HL2, build M_embed
    D_LH = block_dct(LH2)
    D_HL = block_dct(HL2)
    # ECC encode + PRNG
    payload = ecc_encode(W)
    prng = RandomState(seed)
    payload_perm = prng.permutation(payload)
    # Insert spread-spectrum
    for subband in [D_LH, D_HL]:
        for bit_index, bit in enumerate(payload_perm):
            positions = select_R_positions(M_embed, prng, bit_index)
            for pos in positions:
                u,v = pos
                s = prng.choice([-1,1])
                subband[u,v] += alpha_matrix[u%8, v%8] * s * (1 if bit else -1)
    # inverse transforms
    LH2_mod = inverse_block_dct(D_LH)
    HL2_mod = inverse_block_dct(D_HL)
    coeffs['LH2'] = LH2_mod; coeffs['HL2'] = HL2_mod
    I_wm = idwt2(coeffs)
    I_wm = clip_uint8(I_wm)
    # LP_ref compute
    LP_ref = compute_logpolar_fft(I_wm)
    save_state('state.pkl', LP_ref, M_embed, seed, ecc_params, alpha_matrix, tau)
    save_image('groupname imageName.bmp', I_wm)
    return 'groupname imageName.bmp'
```

### Detection (schematic)

```python
def detection(input1, input2, input3):
    I_orig = load_image(input1)
    I_wm = load_image(input2)
    I_att = load_image(input3)
    state = load_state('state.pkl')
    # 1) Fourier-Mellin alignment
    LP_att = compute_logpolar_fft(I_att)
    dx, dy, peak = phase_correlation(state['LP_ref'], LP_att)
    if peak >= corr_thresh:
        scale, theta = decode_scale_rotation(dx, dy)
        I_aligned = warp_image(I_att, scale=scale, rot=theta)
    else:
        # 2) quick non-aligned attempt
        sim_quick = quick_extract_and_compare(I_att, state)
        if sim_quick >= tau_quick:
            return 1, compute_wpsnr(I_wm, I_att)
        # 3) fallback ORB limited with timeout
        I_aligned = orb_ransac_align(I_wm, I_att, timeout=1.0)
        if I_aligned is None:
            return 0, compute_wpsnr(I_wm, I_att)
    # 4) Extract watermark from I_aligned
    W_ex = extract_payload(I_aligned, state)
    sim = normalized_corr(state['W_ref'], W_ex)
    wpsnr = compute_wpsnr(I_wm, I_att)
    decision = 1 if (sim >= state['tau'] and wpsnr >= 35.0) else 0
    return decision, wpsnr
```

---

## 6) Checklist di consegna (file + test)

* `embedding(input1,input2)` correttamente salvato (naming conforme).
* `detection(input1,input2,input3)` rispetta timeout ≤ 5 s.
* `state.pkl` presente e leggero.
* README con parametri usati (seed, alpha_matrix, ecc_params, tau).
* Script ROC (inviato entro deadline) che mostra come è stato scelto `tau`.
* Log di test con esempi di attacchi e risultati (WPSNR e success/fail).

---



# Strategia Ibrida per Watermarking: Robustezza e Velocità in Ambienti a Risorse Limitate

La strategia è ottimizzata per l'ambiente vincolato (3 PC, no server, no GPU) e i tempi stretti ($< 1$ secondo per detection), scartando i modelli complessi di Deep Learning (come VINE o Diffusion) per la loro irrealistica necessità di addestramento. L'approccio si basa su un metodo **ibrido "classico" ottimizzato (DCT/DFT/DWT)**, focalizzato sulla robustezza geometrica (Resizing) e sul massimo **WPSNR** (Weighted Peak Signal-to-Noise Ratio).

***

## 1. Strategia di Embedding (Difesa) 🛡️

L'obiettivo è massimizzare il **WPSNR ($\geq 6$ punti)** e la robustezza contro gli attacchi ristretti, utilizzando un approccio **multi-dominio leggero** ma efficace.

### A. Metodo Ibrido DWT-DCT-SVD (Leggero e Robusto)

Si sfrutta la combinazione di robustezza alla compressione della **DCT**, l'analisi multirisoluzione della **DWT**, e la stabilità della **SVD**.

1.  **DWT (Decomposizione):** Applicare la DWT a un livello sull'immagine in scala di grigi ($I_{gray}$) per ottenere le quattro bande: LL, HL, LH, HH.
2.  **DCT (Localizzazione):** Selezionare la banda **HL** (dettagli orizzontali) o **LH** (dettagli verticali) per l'embedding. Queste bande hanno sufficiente energia per la robustezza ma non troppa per l'invisibilità.
    * Dividere la banda scelta in blocchi $N \times N$ (es., $8 \times 8$).
    * Applicare la DCT a ciascun blocco.
3.  **SVD (Embedding Stabile):** Applicare la SVD ai coefficienti DCT di ciascun blocco. Il watermark **W (1024 bit)** viene incorporato nei **valori singolari ($\Sigma$)**—la firma più stabile della matrice—con un fattore di embedding $\alpha$:
    $$\Sigma' = \Sigma + \alpha \cdot W$$
    dove $\Sigma'$ è la matrice dei valori singolari modificati.

### B. Implementazione Multipla (Redundancy e Forza)

* **Ripetizione (Capacity):** Per aumentare la probabilità di recupero, il watermark di 1024 bit può essere **ripetuto** 2 o 3 volte (ad esempio, incorporando $W_{1}$ in $\text{HL}$ e $W_{2}$ in $\text{LH}$).
* **Forza ($\alpha$) Ottimizzata (HVS):** Utilizzare l'analisi **HVS** (modellabile con la DWT) per calcolare il fattore $\alpha$ più grande possibile che mantenga il **WPSNR $\geq 6$** (6 punti). Un $\alpha$ più grande garantisce una maggiore resistenza a tutti gli attacchi.

***

## 2. Strategia di Detection (Obblighi e Velocità) ⏱️

La funzione di detection deve essere **veloce ($< 1s$)**, **non-blind** (richiede l'immagine originale $I$), e **non deve leggere il file watermark** durante la competizione.

### A. Detection: Non-Blind con Hash/Derivazione

La chiave del metodo è la stabilità della SVD. Il set di valori singolari originali modificati ($\Sigma'$) o un **hash univoco** generato da essi, può essere **hard-coded** nella funzione di detection.

**Procedura (Tempo Reale):**

1.  Estrarre $\Sigma$ e $W'$ (il watermark estratto) con la procedura inversa della SVD e DCT/DWT sull'immagine test $I^*$.
2.  Calcolare la similarità tra il watermark originale $W$ e quello estratto $W'$:
    $$\text{Sim} = \text{Norm}(W, W')$$
3.  La **Detection è fallita** se $\text{Sim} < \tau$ (dove $\tau$ è la soglia ROC) **O** se $\text{WPSNR}(I, I^*) < 5$.

### B. Soglia ROC Ottimale ($\tau$)

* **Calcolo Off-line:** La curva ROC deve essere eseguita **prima della competizione (entro il 27 ottobre)** su un vasto set di immagini, usando **Data Augmentation** che simuli tutti i 6 attacchi permessi.
* **FPR Basso:** Selezionare una soglia $\tau$ che garantisca un **FPR (False Positive Rate) molto basso** ($\leq 0.1\%$). Un FPR basso è fondamentale per evitare la penalità più severa ("rilevare il watermark in immagini non marcate").

***

## 3. Strategia di Attacco (Aggressione Mirata) 💥

L'obiettivo è distruggere il watermark mantenendo il **WPSNR $\geq 5$** per massimizzare il punteggio Quality. L'attacco deve mirare alla **sincronizzazione** e alla probabile banda di embedding (HL/LH).

### A. La Combinazione "Geometrica" Critica

Si predilige la combinazione di attacchi che aggrediscono la sincronizzazione e le bande di frequenza, dato che il **Resizing** è il nemico primario degli schemi DCT/DWT senza rinforzo DFT/SVD.

1.  **Resizing (De-sincronizzazione):**
    * Applicare un ridimensionamento leggero (es. $102\% \rightarrow 98\%$ o $95\% \rightarrow 105\%$).
    * Questo attacco sposta i coefficienti di trasformata e distrugge la sincronizzazione, in particolare negli schemi DCT a blocchi.
2.  **JPEG Compression:**
    * Applicare un Quality Factor (QF) basso (es. **QF = 70**).
    * Questo attacco rimuove i coefficienti di alta e media frequenza, dove il watermark DCT/DWT è tipicamente nascosto.
3.  **Median Filtering:**
    * Utilizzare un kernel **$3 \times 3$ o $5 \times 5$**.
    * Il filtro mediano rimuove il rumore (AWGN) e le micro-modifiche lasciate dal watermark e da altri attacchi, contrastando efficacemente i pattern pseudo-casuali.

### B. Logistica e Velocità

* **Codice Semplice:** Il codice di attacco deve essere una **semplice pipeline di funzioni di OpenCV** (come richiesto) per garantire esecuzione rapida e affidabile con risorse limitate.
* **Log di Attacco:** Registrare con cura l'attacco con il **WPSNR più alto** tra quelli che hanno successo (watermark distrutto) per massimizzare il punteggio di Quality.

# Grey areas 
Sì, le regole della challenge presentano diverse "grey area" (aree grigie) che, se sfruttate strategicamente, possono offrire un vantaggio competitivo, specialmente in un contesto di risorse limitate e vincoli di tempo.

Ecco le principali aree grigie e le strategie per sfruttarle:

---

## 1. Ambito degli Attacchi 💥 (Il Vantaggio dell'Attaccante)

### ⚪ Il *Resizing* come Attacco Geometrico Senza Dati Originali
* [cite_start]**La Regola:** Gli attacchi ammessi includono il **Resizing**[cite: 151]. [cite_start]**NON** è consentito usare l'immagine originale per *localizzare* gli attacchi[cite: 64, 153].
* **L'Area Grigia:** Il *resizing* è l'unico attacco che agisce sulla **sincronizzazione** (geometria) dell'immagine. [cite_start]Sebbene non si possa usare l'originale per *localizzare* il watermark, un attacco di *resizing* (ad esempio, $512 \times 512 \to 500 \times 500 \to 512 \times 512$) **non richiede localizzazione**[cite: 151]. [cite_start]È un attacco globale di de-sincronizzazione[cite: 151]. Molti schemi DCT/DWT classici sono vulnerabili a questo, e non richiede il calcolo di feature complesse.
* **Sfruttamento:** Applicare il *resizing* come primo e più efficace attacco generico. È l'unica opzione geometrica e distrugge la base su cui si poggiano molte implementazioni DCT a blocchi (non invarianti).

### ⚪ L'Attacco Misto con *Sharpening*
* [cite_start]**La Regola:** Gli attacchi ammessi includono lo **Sharpening** (nitidezza)[cite: 149].
* **L'Area Grigia:** Lo *sharpening* è un filtro che enfatizza le alte frequenze. Se combinato con la **Compressione JPEG** (che rimuove le alte frequenze), l'effetto combinato è ambiguo. Può essere usato per **migliorare il WPSNR** dell'immagine attaccata se il watermark aveva leggermente appiattito i dettagli, **oppure** per degradare ulteriormente i coefficienti di alta frequenza usati da schemi deboli di *watermarking*.
* [cite_start]**Sfruttamento:** Usare lo *sharpening* come filtro finale di bilanciamento per aumentare il WPSNR **vicino alla soglia di 35 dB** dopo un attacco distruttivo (es. JPEG QF 70 + Resizing), massimizzando così il punteggio di *Quality*[cite: 183].

---

## 2. Definizione di Successo e Fallimento (Il Vantaggio della Difesa)

### ⚪ Assenza di Dati Originali nella Detection
* [cite_start]**La Regola:** Il codice di *detection* (`detection (input1, input2, input3)`) prende in input l'immagine originale (`input1`), *watermarked* (`input2`), e attaccata (`input3`)[cite: 128, 129, 130]. [cite_start]**NON** deve leggere il file del watermark originale, ma deve estrarre il watermark[cite: 138].
* [cite_start]**L'Area Grigia:** Poiché il codice non deve leggere il file watermark, il modo più semplice è usare il **watermark estratto da `input2`** ($W_{extracted}$ da $I_{watermarked}$) come **referenza** contro cui confrontare $W_{attacked}$ estratto da `input3`[cite: 133]. **Non c'è un requisito esplicito che $I_{original}$ (`input1`) sia indispensabile** nell'algoritmo di detection, a meno che non si stia usando una strategia non-blind pura.
* **Sfruttamento:** Semplificare il codice di *detection* al massimo, concentrandosi sull'estrazione da `input2` e `input3`. [cite_start]Il limite di **5 secondi** [cite: 141] è severo; minimizzare i calcoli (escludendo se possibile il coinvolgimento di `input1` nel calcolo del watermark) è cruciale per la velocità.

### ⚪ La Tolleranza di Robustezza (La Trappola)
* [cite_start]**La Regola:** Un attacco è fallito se il watermark è presente ($sim \ge \tau$) **O** se il WPSNR è $< 35 \text{ dB}$[cite: 140].
* [cite_start]**L'Area Grigia:** L'obiettivo di punteggio *Robustezza* dà **6 punti** se la WPSNR media dell'immagine attaccata con successo è **$< 38 \text{ dB}$**[cite: 179].
* **Sfruttamento:** Progettare l'embedding per essere robusto fino a **$37.9 \text{ dB}$**. Costringendo l'attaccante a spingere l'attacco al punto di distruggere il watermark esattamente a $37.9 \text{ dB}$, si ottiene la massima robustezza (6 punti) e si rende estremamente difficile per l'attaccante raggiungere il punteggio *Quality* (che richiede WPSNR > AVG, che sarà probabilmente sopra i $40 \text{ dB}$). L'attaccante ha una finestra di manovra molto piccola tra i 35 dB e il punto di rottura.

---

## 3. Vincoli di Sottomissione e Codice (Il Vantaggio Logistico)

### ⚪ Fine-Tuning e Sottomissione Tardi
* [cite_start]**La Regola:** Dopo la deadline del 27 ottobre, è concesso solo il **Fine-Tuning** (regolazione di parametri e posizioni di *embedding*)[cite: 36, 37, 83].
* [cite_start]**L'Area Grigia:** Non c'è una definizione rigida di "embedding location" o "parameter"[cite: 37].
* **Sfruttamento:** Sottomettere il codice il 27 ottobre con l'architettura completa, ma con parametri di embedding $\alpha$ (forza) e soglia $\tau$ (detection) che sono **visibilmente ottimizzabili**. Questo permette di "salvare" i miglioramenti di robustezza scoperti all'ultimo minuto da applicare il 3 novembre, come la forza $\alpha$ necessaria per raggiungere esattamente WPSNR $66 \text{ dB}$ o la soglia $\tau$ ottimale.

### ⚪ Tempo di Esecuzione e Test
* [cite_start]**La Regola:** Il codice di *detection* deve completare l'esecuzione in $\le 5$ secondi[cite: 141, 195].
* [cite_start]**L'Area Grigia:** Non è specificato il tempo limite per la funzione di *embedding* e *attack*[cite: 141, 195].
* **Sfruttamento:** L'embedding e l'attacco possono essere più complessi (es. DWT-DCT-SVD per l'embedding), dato che il vincolo di 5 secondi si applica **solo alla *detection***. Concentrare gli sforzi di ottimizzazione della velocità solo su quella funzione (ad esempio pre-calcolando lookup tables o riducendo i loop di SVD/DWT/DCT).


Ottimo — grazie per il testo di Gemini. Farò una **revisione critica puntuale**: prima elenco cosa di quello che ha scritto è **corretto/utile**, poi cosa è **sbagliato/inaspettato o pericoloso** per la gara, e infine propongo **strategie alternative concrete e coerenti con le regole** (con parametri e ottimizzazioni pratiche). Cerco di essere diretto e operativo: niente fronzoli, solo quello che conta per vincere la challenge.

---
# ChatGPT opinion
# 1) Cosa va bene / è plausibile nel testo di Gemini

* **Scelta di rifiutare modelli DL pesanti**: corretta. GPU/tempo/risorse non lo giustificano e il regolamento non lo richiede.
* **Idea di usare un approccio ibrido (DWT/DCT/SVD)**: sensata a livello concettuale — combinare domini spesso migliora la robustezza contro attacchi diversi.
* **Ridondanza e multi-embedding** (ripetere il payload su più bande) e uso di **HVS/masking**: ottime pratiche per avere WPSNR alto e resistenza.
* **Sottolineatura della necessità di ROC e soglia calcolata off-line**: correttissimo e richiesto dal regolamento.
* **Notare che resizing è una minaccia importante per schemi basati su blocchi**: corretto, il resizing rompe sincronizzazione dei blocchi 8×8.

---

# 2) Cosa è sbagliato, fuorviante o rischioso (critiche precise)

1. **Tempo di detection: “<1 s”**

   * Gemini insiste su un requisito di `<1s`. Il regolamento richiede che la detection completi **entro 5 secondi**. Puntare a <1s è ambizioso ma non richiesto; peggio: alcune proposte nel testo (SVD per ogni blocco) rendono difficile rispettare anche i 5s. Corretto: puntare a **≤5 s**, ottimizzando per quello.

2. **Confusione sui WPSNR / punti**

   * Gemini parla di “WPSNR (≥6 punti)” e in vari punti usa soglie sbagliate (es. “WPSNR ≥ 5” — probabilmente intendeva dB ma scritto male). Il file ufficiale usa soglie in **dB** (es. WPSNR ≥ 66 dB dà 6 punti per embedding). Inoltre la condizione per considerare un attacco valido è **WPSNR ≥ 35 dB** (non 5). Bisogna usare numeri esatti.

3. **Uso non chiaro dell’input originale nella detection**

   * Gemini suggerisce che `input1` (originale) potrebbe non essere necessario e che si può confrontare `input2` vs `input3` (estratto da watermarked vs attacked). Il regolamento permette il confronto tra watermark estratto da watermarked e attacked — ma la detection **DEVE** ricevere i tre input e può usare `input1` per calcolare WPSNR. È **consentito** estrarre il riferimento da `input2` (watermarked) e confrontarlo con `input3`. Attenzione: non si può leggere il file watermark originale, ma si può **derivare** o **hashare** valori dalla watermarked image. Quindi la frase di Gemini andava chiarita: non è una "area grigia", è permesso usare `input2` come riferimento ma il detection deve comunque completare entro i limiti e non leggere file esterni.

4. **SVD applicato ai valori singolari di ciascun blocco (Σ' = Σ + α W)** — **problematico**:

   * Tecnica teoricamente usabile, ma **praticamente difficile**:

     * I blocchi 8×8 hanno matrici piccole: SVD su ciascun blocco è costoso e i valori singolari di blocchi piccoli sono **instabili** sotto JPEG/resizing/median.
     * Aggiungere 1024 bit direttamente ai valori singolari richiede **moltissimi blocchi**, mappatura complessa e ridondanza elevata.
     * Eseguire SVD su (512×512)/64 = 4096 blocchi, per più bande e ripetizioni, è **computazionalmente pesante** e rischia di non rientrare nei 5s in detection (o generare WPSNR cattiva se α alto).
   * Conclusione: **SVD su blocchi piccoli è una scelta rischiosa**. Se si usa SVD, meglio applicarla su matrici più grandi (intere subband DWT o patch grandi), o usare versioni approssimate (randomized SVD) ma sempre valutare costo/beneficio.

5. **Richiesta di FPR ≤ 0.1%**

   * Gemini suggerisce FPR estremamente basso (0.1%). Il regolamento chiede di scegliere τ corrispondente a **FPR ∈ [0, 0.1]** — interpretazione ragionevole: 0–0.1 **frazione** = 0–10% (non 0.1%!). Quindi la soglia scelta può essere fino al 10% FPR, non è richiesto 0.1%. Impostare FPR troppo basso peggiora la sensibilità (ossia aumenta i false negatives).

6. **Claim che resizing non richiede localizzazione** ma viene chiamata “area grigia”

   * È vero che resizing è globale e non richiede conoscenza dell’originale. Ma non è un “trucco di regole” — è semplicemente un attacco valido e noto. Non è una scappatoia: la difesa deve prevederlo (sincronizzazione/anchor). Presentarlo come area grigia è fuorviante.

7. **Affermare che si può “salvare” parametri dopo il 27 ottobre**

   * Le regole permettono solo fine-tuning. Non c’è un modo magico per “salvare” architettura completa e poi cambiare strutturalmente il metodo. Se sottoponi un’architettura che richiede dipendenze o binary hidden, rischi penalità. Non sfruttare “ambiguità” qui: rispetta la regola e invia codice eseguibile completo e funzionante.

---

# 3) Strategie alternative concrete e coerenti con le regole (pratiche, rapide, robuste)

Qui propongo **un piano alternativo** (implementabile con OpenCV+NumPy, rispettoso dei limiti computazionali, robusto agli attacchi permessi e veloce nella detection).

## Strategia raccomandata (sintesi)

**Primary:** DCT spread-spectrum su coefficienti mid-frequency di DCT 8×8 con masking HVS.
**Secondary:** DWT multi-scale (2 livelli) embedding leggero sulle bande medie/basse.
**Anchor:** ~150 ORB keypoints per immagine per repliche locali (resilienza a resize).
**Sync:** Piccolo pilot grid (template) in basse frequenze per ri-allineamento (debole, distribuito).
**ECC:** Reed-Solomon o BCH (es. RS(2047,1024) o BCH con t≈100 bit) per recuperare da bit mancanti.
**Detection:** correlazione normalizzata + confronto tra estrazioni da `input2` e `input3` + WPSNR calcolato con `input1`. Ottimizzare in NumPy (nessun SVD per blocco).

### Perché questa combinazione

* Spread-spectrum DCT è **robusto a JPEG, AWGN e sharpening**.
* DWT multi-scale offre resistenza a blur/resize (se si mette ridondanza su scale più basse).
* ORB anchor aiuta a ritrovare regioni dopo resize (feature-based re-alignment).
* ECC consente di ricostruire il payload anche con bit corrotti.
* Nessuna SVD blocco-per-blocco (quindi detection veloce).

---

## Implementazione: dettagli e parametri consigliati

### Embedding (workflow)

1. **Preprocess**: normalizza immagine 512×512 grigio (float32).
2. **ECC**: codifica 1024 bit con RS o BCH → ottieni `payload_bits` di lunghezza n.
3. **DCT layer**:

   * Divide immagine in blocchi 8×8. Per ogni blocco:

     * Applica DCT 2D.
     * Seleziona indici zigzag pos 10–30 (mid-freq).
     * Per ogni bit (sequenza PRNG con seed segreto) applica: `c' = c + α * s`, con `s∈{+1,-1}` pseudo-random per bit.
   * α iniziale: **4.0** (testare e aumentare fino a WPSNR target).
4. **DWT layer**:

   * Applica DWT livello 2 (Haar o db2).
   * Scegli bande medie (LH1/LH2 o HL1) per inserire una versione ridotta del payload con QIM: quant-step q∈{4,6}.
5. **ORB anchor**:

   * Estrai ~150 ORB keypoints su immagine originale.
   * Per i keypoints più stabili, crea piccole patch (32×32) e aplica uno spread-spectrum locale nei coefficienti DCT della patch (α_local = 6 per patch).
6. **Sync template**:

   * Inietta un debole pattern sinusoidale a bassa ampiezza su bassa frequenza (es. tre toni a frequenze radiali) per permettere detection di errore di scala/traslazione minima. Ampiezza molto bassa per non abbassare WPSNR.
7. **Final check**: calcola WPSNR rispetto originale; se < 35 dB rivedi α o la distribuzione.

### Detection (veloce e conforme)

1. **Input**: detection(input1, input2, input3) — leggi le 3 immagini. Calcola WPSNR(watermarked, attacked) da `input2` e `input3`.
2. **Estrai riferimento**: estrai `payload_ref` da `input2` (stesso processo di estrazione: DWT+DCT local + ORB patches).
3. **Estrai attaccato**: estrai `payload_att` da `input3`.
4. **Decodifica ECC**: tenta la correzione. Calcola percentuale bit uguali.
5. **Similitudine**: `sim = corr(payload_ref, payload_att)` oppure Hamming distance dopo ECC.
6. **Decisione**: se `sim >= τ` → watermark presente (output1=1), altrimenti 0. Inoltre output2 = WPSNR(`input2`,`input3`).

   * τ scelto off-line tramite ROC con FPR nel range [0,0.1] (0–10%). **Non** fissare 0.1% come Gemini suggeriva.

**Ottimizzazioni per velocità (necessarie per ≤5s)**:

* Vectorizza trasformate: usa `scipy.fftpack.dct` su tutta la immagine a blocchi tramite reshape (evitare loop Python).
* Evita SVD per blocco: SVD solo su subband globale se strettamente necessario.
* Precalcola maschere e indici e caricale come costanti.
* Limitare numero di keypoint patches a 100–150 per immagine (estrazione ORB è veloce).

---

## Difesa contro resizing (la minaccia principale)

* **Anchor features (ORB)**: usale come punti di replica; in detection cerca corrispondenze e ri-allinea le patch tramite similitudine affina (estimare scala e traslazione grossolana usando orb+RANSAC).
* **Pilot tones a basse frequenze**: per stimare shift/scale globale via phase correlation su subband a bassa frequenza.
* **Inserisci ridondanza del payload su scale basse DWT**: anche se blur/median eliminano certe componenti, le scale basse rimangono.

---

## Contromisure all’attacco “resizing + JPEG + median” suggerito da Gemini

* Gli attacchi misti proposti (resize → JPEG QF70 → median → sharpen) sono efficaci contro schemi deboli DCT-only. Contromisura:

  * mettere più ridondanza sulle bande basse (DWT), usare anchor ORB per riallineamento, e usare ECC forte.
  * attenzione a non aumentare α eccessivamente (riduce WPSNR).

---

# 4) Checklist pratica e priorità di sviluppo (per il tuo gruppo)

1. **Implementare prototype DCT spread-spectrum** (embedding + extraction) — test baseline.
2. **Aggiungere ECC** e valutare ROC su dataset sintetico con attacchi predefiniti.
3. **Aggiungere DWT multi-scale** (secondo layer) e verificare extra robustezza vs blur/resize.
4. **Aggiungere ORB anchors** per resync (test resize 0.5–1.5).
5. **Ottimizzare detection** per essere <5s (vectorize, ridurre SVD).
6. **Generare ROC e scegliere τ** (FPR tra 0% e 10%; preferire 1–5% per bilanciamento).
7. **Test largo**: AWGN σ∈{2,4,8}, JPEG QF∈{50,70,90}, median 3/5/7, resize {0.5,0.75,1.5} + ritorno alla 512, blur σ∈{0.5,1,2}.

---

# 5) Conclusione sintetica (cosa accettare di Gemini e cosa cambiare subito)

* **Accetta**: approccio ibrido, ridondanza, fine-tuning via ROC, attenzione a resizing.
* **Cambia/Substituisci**: evita SVD su blocchi 8×8 come meccanismo principale; non inseguire FPR 0.1% (troppo stringente); non assumere requisito <1s per detection; non confondere le soglie WPSNR (usa le soglie del regolamento: 35 dB min per considerare attacco valido, ecc.).
* **Sostituzione pratica**: uso DCT spread-spectrum + DWT + ORB anchors + ECC + sync pilot è una soluzione più pratica, rapida ed efficace rispetto alla DWT-DCT-SVD per blocco.

---

Se vuoi, procedo subito con **una cosa pratica** (scegli una sola opzione e la faccio subito nel mio prossimo messaggio — non posso lavorare in background):

1. Boilerplate Python minimal (embedding/detection skeleton) che implementa **DCT spread-spectrum + ECC + ROC test harness**; o
2. Script di test che genera dataset, applica attacchi permessi (AWGN, JPEG, median, resize, blur, sharpen) e costruisce la ROC per scegliere τ; o
3. Codice di detection altamente ottimizzato (senza SVD) che rispetta i 5s e usa estrazione correlazionale.

Dimmi quale preferisci e te lo preparo immediatamente.

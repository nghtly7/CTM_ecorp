
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
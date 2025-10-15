
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
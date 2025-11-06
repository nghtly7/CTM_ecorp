# Multimedia Data Security - Watermarking Challenge

~~~
Project for the Multimedia Data Security course at the University of Trento.
~~~

* **Group:** ecorp
* **Members:**
    * Nicolò Fadigà
    * Davide Martini
    * Alberto Messa
    * Francesco Poinelli

## Challenge Overview
The challenge involved developing a robust image watermarking technique and its corresponding detection function. Key tasks included analyzing the technique's performance via a ROC curve and engineering an attack strategy to compromise watermarks implemented by other groups.

## How to Test

### Prerequisites

1.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
2.  Navigate to the source directory:
    ```bash
    cd src/
    ```
3.  **Important:** Check the configuration parameters (e.g., folder paths, filenames) at the beginning of each Python script before running.

### Embedding & ROC Curve

* **Watermark images:**
    ```bash
    python run_embedding.py
    ```
* **Generate and evaluate the ROC curve:**
    ```bash
    python ROC_ecorp.py
    ```

### Attack Strategy

1.  **Setup:** Create a `groups/` directory in `src/`. Inside `groups/`, create subfolders for each target group (e.g., `groups/group_A/`). Place their watermarked images and detection function inside their respective folder.

2.  **Run Attacks:**
    * **Brute-force attack (on a specific group):**
        ```bash
        python attack_brute_force.py
        ```
    * **Test a single attack (on a single image):**
        ```bash
        python attack_manual.py
        ```
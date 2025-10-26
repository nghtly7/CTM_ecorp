# Attack Strategy: Parallel Brute-Force

## Purpose

This script automatically tests a large set of attacks (single and paired) to find a combination that removes an image watermark. The goal is to find an attack that successfully removes the mark (`found == 0`) while maintaining the highest possible quality (**WPSNR $\ge$ 35 dB**).

## Usage

1.  Configure the `ADV_GROUP_NAME` (the group you are attacking) at the top of the script.
2.  Run the script:
    ```bash
    python attack_brute_force.py
    ```

## Execution Strategy: Parallel by Type

This script is optimized for speed using a "Parallel by Type" approach.

1.  **Attack Generation:** It creates a list of all single and paired attacks, ordered from least to most aggressive.
2.  **Smart Grouping:** Attacks are grouped into "jobs" by type. Each job represents a list of attacks for one worker to test sequentially.
      * ***Single Attack Jobs:*** Groups all parameter variations for a single attack. The worker iterates through these parameters.
          * *Example Job:* `('jpeg',)` contains the list: `[jpeg(50), jpeg(45), jpeg(40)...]`
      * ***Paired Attack Jobs:*** Groups attacks by fixing the *first attack and its parameter*, then iterating through all parameters of the *second* attack.
          * *Example Job:* `('jpeg', '[50]', 'blur')` contains the list: `[jpeg(50)+blur(0.4), jpeg(50)+blur(0.8), jpeg(50)+blur(1.2)...]`

      Since the master list (from step 1) is already sorted, this grouping guarantees that every worker iterates through its job list from the softest attack to the most aggressive.
3.  **Worker Pool:** The script assigns each "job" to a different CPU worker (process). This allows it to test `jpeg`, `blur`, and `jpeg+blur` all simultaneously.
4.  **Optimized Worker Logic:** Each worker tests its assigned job (list) sequentially and uses "early stopping" for efficiency:
      * **Early Success:** If an attack is successful (`found == 0` & `WPSNR >= 35`), the worker **stops immediately** and returns that result (as it's the first success in that ordered group).
      * **Early Failure (Too Strong):** If an attack fails the quality check (`WPSNR < 35`), the worker **stops immediately**, skipping all stronger, subsequent attacks in its job.
5.  **Final Result:** The main script collects all successful results from the workers and selects the single best one (the attack with the **highest WPSNR**).
6.  **Output:**
      * **Image:** Saves the final attacked image *only* if a successful attack was found, placing it in `attack_results/`.
      * **CSV Log:** It *always* writes one row to `attack_results/result.csv` for *every* image processed, logging either the successful attack details or "NO\_SUCCESSFUL\_ATTACK\_FOUND".
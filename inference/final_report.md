# Market Regime Detection System — Final Report

Generated on: 2026-04-22 23:06:16

## System Architecture
- LSTM BiLSTM encoder (167,427 params, frozen)
- Modern Hopfield BAM module (4,676 params)
- Markov-Viterbi smoother + VIX hybrid override
- Three regimes: Growth, Transition, Panic

## Training Results
- LSTM best val accuracy: 71.86%
- LSTM test accuracy: 65.43%
- Training stopped at epoch 11 (early stopping)

## BAM Results
- BAM best val accuracy: 71.73%
- BAM test accuracy: 65.69%

## Final Pipeline Results
- Overall accuracy: 70.1%
- Growth F1: 0.764, Recall: 0.925, Precision: 0.650
- Transition F1: 0.618, Recall: 0.507, Precision: 0.790
- Panic F1: 0.679, Recall: 0.600, Precision: 0.783
- Panic recall improvement: 0% -> 60%

## Walk-Forward Results
- 2022: 48.61% (bear market year)
- 2023: 76.00% (recovery year)
- 2024: 85.66% (bull market year)
- Average: 70.09%

## Key Findings
- The system performs best in trending markets
- Panic detection relies on VIX hybrid override
- BAM attention correctly maps regimes to prototypes
- Overfitting was observed due to limited dataset size

## Limitations
- Dataset size: 3,805 training samples
- Panic class imbalance: only 346 training samples
- Test period limited to 2022-2024

## Files Generated
- models/checkpoints/lstm_best.pt
- models/checkpoints/bam_best.pt
- models/checkpoints/markov_params.npz
- models/checkpoints/training_history.json
- models/checkpoints/bam_history.json
- models/checkpoints/final_pipeline_results.json
- models/checkpoints/walk_forward_results.csv
- models/checkpoints/walk_forward_performance.png
- models/checkpoints/walk_forward_regime_calendar.png
- models/checkpoints/markov_smoothing_timeline.png
- models/checkpoints/markov_confusion_comparison.png
- inference/predict.py
- inference/alerts.py
- inference/alerts.json
- inference/final_report.md
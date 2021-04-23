
**Metric Options

***Comparison types

There are two groups of comparison types:

*1. Matching*

These are the classic few shot accuracy calculations, where for each sample in a study (or recall set), we find the closest sample in the recall (or study) set.
If the closest one is the correct label, then it is considered to be correctly classified.
The 'closest' is determined using different options, given below:

- match_mse - mean square error
- match_overlap - count of bits that are equal (assumes binary) (`sum(a*b)`)
- match_cos - cosine similarity

*2. Loss*

This is the sum of an element-wise loss calculation.
For example, the mse difference between the *i*'th study sample, and the *i*'th recall sample, for all *i*.

The comparison types are:

- accuracy - average number that are exactly equal
- mse - mean square error
- mismatch - mean of absolute error
- cos - cosine similarity

***Features

For the AHA STM, the possible features are:

- stm_ps
- stm_pr
- stm_pc
- stm_recon
- stm_recon_ec


**Specifying Metrics

It is possible to add metrics in code or in the json.
You select a text label (just for display purposes), comparison type, the metric, and primary/secondary.
The definition of primary/secondary is below.
In matching, for each sample in the **primary** set (study or recall), we find the closest sample in the other **secondary** set.
The choice does make a difference.

Here is an example of two metrics being added via json:
- The first is ltm matching accuracy calculated using MSE (with study being the primary feature, indicated with sf = 'study first'), and
- The second is AHA's PR loss, also calculated with MSE.

```
  "metrics": {
    "prefixes": ["ltm_sf", "pr"],
    "primary_feature_names": ["study.ltm","recall.stm_pr"],
    "primary_label_names": ["study.labels", "recall.labels"],
    "secondary_feature_names": ["recall.ltm","study.stm_pr"],
    "secondary_label_names": ["recall.labels", "study.labels"],
    "comparison_types": ["match_mse", "mse"]
  }
```

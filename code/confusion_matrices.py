import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# File paths
data_path = (
    "./taxonomy/CLA_Taxonomy_Validation_Data.csv"
)

png_path = (
    "./Plots/EAA_AEAA_Confusion_Matrices.png"
)


# Load raw annotation data

df_raw = pd.read_csv(data_path)

# Remove accidental leading/trailing whitespace from column names
df_raw.columns = df_raw.columns.str.strip()

# Rename columns to simpler analysis names
df_raw = df_raw.rename(columns={
    "PROLIFIC_PID": "PID",
    "Duration (in seconds)": "Duration"
})


# Check raw data
print("=" * 70)
print("Raw annotation data")
print("=" * 70)

print(f"Annotations:  {len(df_raw):,}")
print(f"Participants: {df_raw['PID'].nunique():,}")
print(f"Items:        {df_raw['Item'].nunique():,}")

print("\nColumns:")
# print(df_raw.columns.tolist())

# participant-level quality control
#
# Participants are excluded if:
#
#   1. They fail the attention check
#
#   or
#
#   2. Their completion time falls within the fastest
#      10% of participants.
#
# Exclusion occurs at the participant level, meaning all
# annotations from an excluded participant are removed.

# Create participant-level QC table
participant_qc = (
    df_raw
    .groupby("PID")
    .agg(
        Duration=("Duration", "mean"),
        Attention=("Attention", "first")
    )
    .reset_index()
)

# Participants who failed the attention check
failed_attention_ids = set(
    participant_qc.loc[
        participant_qc["Attention"] != "No",
        "PID"
    ]
)

# Determine 10th percentile completion-time threshold
duration_threshold = participant_qc["Duration"].quantile(0.10)

# Participants in the fastest 10%
fast_ids = set(
    participant_qc.loc[
        participant_qc["Duration"] <= duration_threshold,
        "PID"
    ]
)

# Combine participant-level exclusion criteria
bad_ids = failed_attention_ids.union(fast_ids)

# Apply QC exclusions
df = (
    df_raw.loc[
        ~df_raw["PID"].isin(bad_ids)
    ]
    .copy()
)

# QC summary
print("\n" + "=" * 70)
print("Participant-level quality control")
print("=" * 70)

print(
    f"10th percentile duration threshold: "
    f"{duration_threshold:.4f} seconds"
)

print(
    f"Participants failing attention check: "
    f"{len(failed_attention_ids)}"
)

print(
    f"Participants in fastest 10%: "
    f"{len(fast_ids)}"
)

print(
    f"Unique participants excluded: "
    f"{len(bad_ids)}"
)


print("\nPost-QC analytic data")

print(f"Annotations:  {len(df):,}")
print(f"Participants: {df['PID'].nunique():,}")
print(f"Items:        {df['Item'].nunique():,}")

# category definitions
actual_categories = [
    "Code-Mixing",
    "Loan Words",
    "Sheng",
    "Tribal Lexicons"
]

# EAA permits Standard Swahili as the ensemble prediction
eaa_prediction_categories = [
    "Code-Mixing",
    "Loan Words",
    "Sheng",
    "Tribal Lexicons",
    "Standard Swahili"
]

# AEAA excludes Standard Swahili before majority voting
aeaa_prediction_categories = [
    "Code-Mixing",
    "Loan Words",
    "Sheng",
    "Tribal Lexicons"
]

# Deterministic majority-vote function
#
# For each item:
#
#   1. Count the number of annotations assigned to each label.
#   2. Identify the label(s) receiving the maximum count.
#   3. If there is a tie, sort tied labels alphabetically and
#      select the first one.
#
# This makes the result deterministic and independent of
# dataframe row order.

def majority_vote(series):

    counts = series.value_counts()

    max_count = counts.max()

    winners = sorted(
        counts[
            counts == max_count
        ].index.astype(str)
    )

    return winners[0]

# Determine actual category for each item
item_actual = (
    df[
        ["Item", "Actual"]
    ]
    .drop_duplicates(subset="Item") 
    .set_index("Item")["Actual"]
)

# Ensemble Annotation Accuracy (EAA)

# EAA uses the majority annotation for each item across all
# available annotation categories, including Standard Swahili.

# Compute item-level majority prediction
eaa_predictions = (
    df
    .groupby("Item")["Response"]
    .apply(majority_vote)
)

# combine actual and ensemble-predicted categories
eaa_results = pd.concat(
    [
        item_actual.rename("Actual"),
        eaa_predictions.rename("Predicted")
    ],
    axis=1
).dropna()

# retain the four CLA ground-truth categories
eaa_results = (
    eaa_results.loc[
        eaa_results["Actual"].isin(actual_categories)
    ]
    .copy()
)

# determine whether ensemble prediction is correct
eaa_results["Correct"] = (
    eaa_results["Actual"]
    ==
    eaa_results["Predicted"]
)

# overall EAA
overall_eaa = eaa_results["Correct"].mean()

# Category-level EAA
category_eaa = (
    eaa_results
    .groupby("Actual")["Correct"]
    .mean()
    .reindex(actual_categories)
)

# Print EAA results
print("\n" + "=" * 70)
print("Ensemble Annotation Accuracy (EAA)")
print("=" * 70)

print(f"Overall EAA: {overall_eaa:.4f}")

print("\nCategory-level EAA:")

for category, value in category_eaa.items():

    print(
        f"{category:<20} "
        f"{value:.4f}"
    )

# EAA Confusion matrix
eaa_counts = pd.crosstab(
    eaa_results["Actual"],
    eaa_results["Predicted"]
)

# Force consistent row/column ordering
eaa_counts = eaa_counts.reindex(
    index=actual_categories,
    columns=eaa_prediction_categories,
    fill_value=0
)

# Row-normalize the confusion matrix
cm_eaa_norm = eaa_counts.div(
    eaa_counts.sum(axis=1),
    axis=0
)


print("\nEAA Confusion Matrix — Counts")

print(
    eaa_counts
)


print("\nEAA Confusion Matrix — Row-Normalized")

print(
    cm_eaa_norm.round(4)
)


# Adjusted Ensemble Annotation Accuracy (AEAA)

# Note: AEAA removes annotations labeled "Standard Swahili"
# before calculating the item-level majority vote.

# Remove Standard Swahili annotations
df_aeaa = (
    df.loc[
        df["Response"] != "Standard Swahili"
    ]
    .copy()
)

# compute adjusted item-level majority prediction
aeaa_predictions = (
    df_aeaa
    .groupby("Item")["Response"]
    .apply(majority_vote)
)

# combine actual and adjusted predictions
aeaa_results = pd.concat(
    [
        item_actual.rename("Actual"),
        aeaa_predictions.rename("Predicted")
    ],
    axis=1
).dropna()

# retain the four CLA ground-truth categories
aeaa_results = (
    aeaa_results.loc[
        aeaa_results["Actual"].isin(actual_categories)
    ]
    .copy()
)

# determine whether adjusted ensemble prediction is correct
aeaa_results["Correct"] = (
    aeaa_results["Actual"]
    ==
    aeaa_results["Predicted"]
)

# overall AEAA
overall_aeaa = aeaa_results["Correct"].mean()

# category-level AEAA
category_aeaa = (
    aeaa_results
    .groupby("Actual")["Correct"]
    .mean()
    .reindex(actual_categories)
)

# print AEAA results
print("\n" + "=" * 70)
print("Adjusted Ensemble Annotation (AEAA)")
print("=" * 70)

print(f"Overall AEAA: {overall_aeaa:.4f}")

print("\nCategory-level AEAA:")

for category, value in category_aeaa.items():

    print(
        f"{category:<20} "
        f"{value:.4f}"
    )

# AEAA confusion matrix

aeaa_counts = pd.crosstab(
    aeaa_results["Actual"],
    aeaa_results["Predicted"]
)

# force consistent row/column ordering
aeaa_counts = aeaa_counts.reindex(
    index=actual_categories,
    columns=aeaa_prediction_categories,
    fill_value=0
)

# row-normalize the confusion matrix
cm_aeaa_norm = aeaa_counts.div(
    aeaa_counts.sum(axis=1),
    axis=0
)


print("\nAEAA Confusion Matrix — Counts")

print(
    aeaa_counts
)


print("\nAEAA Confusion Matrix — Row-Normalized")

print(
    cm_aeaa_norm.round(4)
)

#reproducibility checks:

# These are the canonical confusion-matrix counts used for
# the manuscript figure.
#
# If the raw data or analysis procedure changes in a way that
# changes the figure, the script will stop here

expected_eaa_counts = np.array([
    [57, 4, 0, 0, 1],
    [2, 61, 0, 0, 9],
    [0, 0, 4, 0, 1],
    [0, 0, 0, 27, 4]
])


expected_aeaa_counts = np.array([
    [57, 4, 0, 1],
    [2, 68, 0, 0],
    [0, 0, 5, 0],
    [0, 0, 0, 31]
])


assert np.array_equal(
    eaa_counts.values,
    expected_eaa_counts
), (
    "EAA confusion matrix does not match the "
    "canonical manuscript results."
)


assert np.array_equal(
    aeaa_counts.values,
    expected_aeaa_counts
), (
    "AEAA confusion matrix does not match the "
    "canonical manuscript results."
)


print("\n" + "=" * 70)
print("Reproducibility check passed")
print("=" * 70)

print(
    "EAA and AEAA confusion matrices match "
    "the canonical manuscript results."
)

# prepare normalized matrices for plotting
data_eaa = cm_eaa_norm.loc[
    actual_categories,
    eaa_prediction_categories
].values


data_aeaa = cm_aeaa_norm.loc[
    actual_categories,
    aeaa_prediction_categories
].values

# create two-panel figure
fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 7)
)

# panel A: EAA
ax = axes[0]


im1 = ax.imshow(
    data_eaa,
    cmap="Blues",
    vmin=0,
    vmax=1,
    aspect="auto"
)

# axis ticks
ax.set_xticks(
    np.arange(
        len(eaa_prediction_categories)
    )
)

ax.set_yticks(
    np.arange(
        len(actual_categories)
    )
)

# axis tick labels
ax.set_xticklabels(
    eaa_prediction_categories,
    rotation=35,
    ha="right",
    fontsize=11
)

ax.set_yticklabels(
    actual_categories,
    fontsize=11
)

# axis labels
ax.set_xlabel(
    "Predicted",
    fontsize=12
)

ax.set_ylabel(
    "Actual",
    fontsize=12
)

# panel title
ax.set_title(
    "(a) Ensemble Annotation Accuracy (EAA)",
    fontsize=13,
    fontweight="bold",
    pad=12
)

# add normalized values to cells
for i in range(data_eaa.shape[0]):

    for j in range(data_eaa.shape[1]):

        value = data_eaa[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=11,
            color=(
                "white"
                if value > 0.50
                else "black"
            )
        )

# panel B: AEAA
ax = axes[1]

im2 = ax.imshow(
    data_aeaa,
    cmap="Blues",
    vmin=0,
    vmax=1,
    aspect="auto"
)

# axis ticks
ax.set_xticks(
    np.arange(
        len(aeaa_prediction_categories)
    )
)

ax.set_yticks(
    np.arange(
        len(actual_categories)
    )
)

# axis tick labels
ax.set_xticklabels(
    aeaa_prediction_categories,
    rotation=35,
    ha="right",
    fontsize=11
)

ax.set_yticklabels(
    actual_categories,
    fontsize=11
)

# axis labels
ax.set_xlabel(
    "Predicted",
    fontsize=12
)

ax.set_ylabel(
    "Actual",
    fontsize=12
)

# panel title
ax.set_title(
    "(b) Adjusted Ensemble Annotation Accuracy (AEAA)",
    fontsize=13,
    fontweight="bold",
    pad=12
)

# add normalized values to cells
for i in range(data_aeaa.shape[0]):

    for j in range(data_aeaa.shape[1]):

        value = data_aeaa[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=11,
            color=(
                "white"
                if value > 0.50
                else "black"
            )
        )

# panel spacing
plt.subplots_adjust(
    left=0.08,
    right=0.88,
    bottom=0.28,
    top=0.86,
    wspace=0.35
)


# shared colorbar:
# align the colorbar exactly with the height of the
# right-hand confusion matrix.

fig.canvas.draw()


# position of AEAA matrix
pos = axes[1].get_position()


# colorbar axis
cbar_ax = fig.add_axes([
    pos.x1 + 0.01,
    pos.y0,
    0.015,
    pos.height
])


cbar = fig.colorbar(
    im2,
    cax=cbar_ax
)


cbar.set_label(
    "Proportion",
    fontsize=12
)


cbar.ax.tick_params(
    labelsize=10
)

# save figure

plt.savefig(
    png_path,
    dpi=1000,
    bbox_inches="tight"
)
# display

plt.show()

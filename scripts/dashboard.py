import streamlit as st
import pandas as pd

st.markdown(
    """
<style>
section.main > div{
max-width: 1500px !important;
margin: 30px 100px;
}
</style>
""",
    unsafe_allow_html=True,
)

if __name__ == '__main__':
    artists = pd.read_csv("data/artists.csv")
    tracks = pd.read_csv("data/tracks.csv")

    q1 = pd.read_csv("output/q1.csv")
    q2 = pd.read_csv("output/q2.csv")
    q3 = pd.read_csv("output/q3.csv")

    cv_gbt_config = pd.read_csv("output/cv_gbt_config.csv")
    best_gbt_params = pd.read_csv("output/best_gbt_params.csv")
    best_gbt_scores = pd.read_csv("output/best_gbt_scores.csv")

    cv_rf_config = pd.read_csv("output/cv_rf_config.csv")
    best_rf_params = pd.read_csv("output/best_rf_params.csv")
    best_rf_scores = pd.read_csv("output/best_rf_scores.csv")
    rf_features = pd.read_csv("output/rf_features.csv")

    rf_popularity = pd.read_json("output/rf_popularity.json", lines=True)
    gbt_popularity = pd.read_json("output/gbt_popularity.json", lines=True)

    with open('output/rf_time.txt', 'r') as f:
        rf_perf = f.read()
    with open('output/gbt_time.txt', 'r') as f:
        gbt_perf = f.read()

    st.title("Big Data Project **2023**")
    st.write(
        "Panov Evgenii (e.panov@innopolis.university)"
    )

    st.markdown("---")
    st.header("Descriptive Data Analysis")
    st.subheader("Data Characteristics")
    general_dda = pd.DataFrame(
        data=[
            ["Artists", artists.shape[0] - 1, artists.shape[1]],
            ["Tracks", tracks.shape[0], tracks.shape[1]],
        ],
        columns=["Table", "Instances", "Features"],
    )
    st.write(general_dda)

    st.markdown("`artists` table")
    st.write(artists.describe())

    st.markdown("`tracks` table")
    st.write(tracks.describe())

    st.subheader("Some samples from the data")
    st.markdown("`artists` table")
    st.write(artists.head(5))

    st.markdown("`tracks` table")
    st.write(tracks.head(5))

    st.markdown("---")
    st.header("Exploratory Data Analysis")
    st.subheader("Q1")
    st.markdown("Correlation between popularity and the track's release year")
    st.write(q1)
    st.markdown(
        "As we can see, the correlation is relatively high, so we want to use it as a feature"
    )

    st.subheader("Q2")
    st.markdown("How many danceable tracks are in the most popular ones?")
    st.write(q2)
    st.markdown(
        "As we can see, most of the popular tracks and danceable as well. So this parameter is very useful for us"
    )

    st.subheader("Q3")
    st.markdown(
        "Correlation between popularity and instrumentalness"
    )
    st.write(q3)
    st.markdown(
        "As we can see, there is no correlation between popularity and instrumentalness, which means that such feature doesn't exist"
    )

    st.markdown("---")
    st.header("Predictive Data Analytics")
    st.subheader("ML Models")
    st.markdown("#### 1. GBT Model")
    st.markdown("- Cross validation config")
    st.write(cv_gbt_config)
    st.markdown("- Best model parameters")
    st.write(best_gbt_params)
    st.markdown("- Best model scores")
    st.write(best_gbt_scores)
    st.markdown("- Model time taken")
    st.write(gbt_perf)
    st.markdown("- Popularity example")
    st.write(gbt_popularity.head(20))

    st.markdown("#### 2. Random Forest Model")
    st.markdown("- Columns used as features")
    st.write(rf_features)
    st.markdown("- Cross validation config")
    st.write(cv_rf_config)
    st.markdown("- Best model parameters")
    st.write(best_rf_params)
    st.markdown("- Best model scores")
    st.write(best_rf_scores)
    st.markdown("- Model time taken")
    st.write(rf_perf)
    st.markdown("- Popularity example")
    st.write(rf_popularity.head(20))

    st.markdown("---")
    st.subheader("Results")
    st.markdown(
        "We used the following metrics for estimating the performance of our models"
    )
    st.markdown(
        "- `RMSE` is a measure of the accuracy of a predictive model. It stands for Root Mean Square Error and the formula is simple - sqrt(sum((x_i - y_i) ^ 2) / n)"
    )
    st.markdown(
        "- `R^2` indicates how well the model captures and accounts for the variation in the data."
    )
    st.markdown(
        "As you can see GBT performed a bit better than RF for ~4.5%. I think because Gradient Boosting optimizes the model parameters using gradient descent, which can help it find a more precise and efficient path to the optimal solution compared to the independent tree construction of Random Forests."
    )

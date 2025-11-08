import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

@st.cache_data
def load_data(path="Automobile.csv"):
    df = pd.read_csv(path)
    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]
    # Convert numeric columns which may have missing values
    df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
    df['mpg'] = pd.to_numeric(df['mpg'], errors='coerce')
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
    df['cylinders'] = pd.to_numeric(df['cylinders'], errors='coerce')
    df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')
    df['displacement'] = pd.to_numeric(df['displacement'], errors='coerce')
    df['acceleration'] = pd.to_numeric(df['acceleration'], errors='coerce')
    # Drop rows with missing mpg or weight for many visuals, but keep in general df
    return df

df = load_data()

st.title("Automobile Explorer — Interactive Visualizations and MPG Predictor")
st.markdown("Use the controls on the left to filter the dataset. Visualizations update automatically. Includes a simple linear model to predict MPG from weight and horsepower.")

# Sidebar - Filters
st.sidebar.header("Filter dataset")

# Origin filter
origins = sorted(df['origin'].dropna().unique().tolist())
selected_origins = st.sidebar.multiselect("Origin", options=origins, default=origins)

# Cylinders filter
cyls = sorted(df['cylinders'].dropna().unique().astype(int).tolist())
selected_cyls = st.sidebar.multiselect("Cylinders", options=cyls, default=cyls)

# Year range
min_year = int(df['model_year'].min())
max_year = int(df['model_year'].max())
year_range = st.sidebar.slider("Model year range", min_value=min_year, max_value=max_year, value=(min_year, max_year), step=1)

# Horsepower slider (handle NaNs)
hp_min = int(np.nanmin(df['horsepower']))
hp_max = int(np.nanmax(df['horsepower']))
hp_range = st.sidebar.slider("Horsepower range", min_value=hp_min, max_value=hp_max, value=(hp_min, hp_max))

# Weight slider
w_min = int(df['weight'].min())
w_max = int(df['weight'].max())
weight_range = st.sidebar.slider("Weight range", min_value=w_min, max_value=w_max, value=(w_min, w_max))

# Text search on name
name_search = st.sidebar.text_input("Search model name (contains)")

# Checkbox: show only complete rows for modeling
only_complete = st.sidebar.checkbox("Only rows with horsepower & weight (for some visuals / model)", value=True)

# Apply filters
filtered = df.copy()
filtered = filtered[filtered['origin'].isin(selected_origins)]
filtered = filtered[filtered['cylinders'].isin(selected_cyls)]
filtered = filtered[(filtered['model_year'] >= year_range[0]) & (filtered['model_year'] <= year_range[1])]
filtered = filtered[(filtered['horsepower'] >= hp_range[0]) & (filtered['horsepower'] <= hp_range[1])]
filtered = filtered[(filtered['weight'] >= weight_range[0]) & (filtered['weight'] <= weight_range[1])]
if name_search:
    filtered = filtered[filtered['name'].str.contains(name_search, case=False, na=False)]

if only_complete:
    filtered_complete = filtered.dropna(subset=['horsepower', 'weight', 'mpg'])
else:
    filtered_complete = filtered

st.subheader(f"Filtered dataset — {len(filtered)} rows")
st.dataframe(filtered[['name','mpg','cylinders','horsepower','weight','model_year','origin']].reset_index(drop=True))

# Visual 1: Scatter MPG vs Weight colored by origin, size by horsepower
st.subheader("MPG vs Weight (interactive)")
if len(filtered_complete) >= 2:
    fig_scatter = px.scatter(
        filtered_complete,
        x='weight',
        y='mpg',
        color='origin',
        size='horsepower',
        hover_data=['name','model_year','cylinders'],
        labels={'weight':'Weight','mpg':'MPG'}
    )
    # Add a simple regression line computed over filtered_complete
    try:
        # dropna for regression
        reg_df = filtered_complete.dropna(subset=['weight','mpg'])
        x = reg_df['weight'].values.reshape(-1,1)
        y = reg_df['mpg'].values
        lr = LinearRegression()
        lr.fit(x, y)
        xs = np.linspace(reg_df['weight'].min(), reg_df['weight'].max(), 100)
        ys = lr.predict(xs.reshape(-1,1))
        fig_scatter.add_traces(px.line(x=xs, y=ys, labels={'x':'weight','y':'mpg'}).data)
        fig_scatter.update_layout(legend_title_text='Origin')
    except Exception as e:
        st.write("Could not compute regression line:", e)
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.write("Not enough complete rows to make scatter plot after filtering.")

# Visual 2: Histogram of horsepower
st.subheader("Horsepower distribution")
if len(filtered['horsepower'].dropna()) > 0:
    fig_hp = px.histogram(filtered, x='horsepower', nbins=25, title="Horsepower distribution")
    st.plotly_chart(fig_hp, use_container_width=True)
else:
    st.write("No horsepower data available for this filter.")

# Visual 3: Boxplot MPG by cylinders
st.subheader("MPG distribution by cylinder count")
if 'cylinders' in filtered.columns and len(filtered.dropna(subset=['mpg','cylinders']))>0:
    fig_box = px.box(filtered, x='cylinders', y='mpg', points='outliers', title="MPG by cylinders")
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.write("Insufficient data for boxplot.")

# Visual 4: Average MPG by model_year
st.subheader("Average MPG by model year")
if len(filtered.dropna(subset=['mpg','model_year']))>0:
    mpg_by_year = filtered.groupby('model_year', as_index=False)['mpg'].mean().sort_values('model_year')
    fig_line = px.line(mpg_by_year, x='model_year', y='mpg', markers=True, title="Average MPG by model year")
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.write("No MPG/year data for this filter.")

# Visual 5: Correlation heatmap (numeric cols)
st.subheader("Correlation heatmap (numeric features)")
num_cols = filtered.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols) >= 2:
    corr = filtered[num_cols].corr()
    fig_heat = px.imshow(corr, text_auto=True, aspect="auto", title="Feature correlation")
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.write("Not enough numeric columns for correlation heatmap.")

# Simple predictive model: predict MPG from weight and horsepower
st.subheader("Simple MPG predictor (Linear Regression)")

modelable = df.dropna(subset=['weight','horsepower','mpg'])
if len(modelable) < 10:
    st.write("Not enough complete rows to train a model.")
else:
    X = modelable[['weight','horsepower']].values
    y = modelable['mpg'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    y_pred = lm.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Trained linear regression to predict MPG from weight and horsepower. Test R² = {r2:.3f}")

    st.markdown("Enter weight and horsepower to get a predicted MPG:")
    inp_weight = st.number_input("Weight", min_value=int(df['weight'].min()), max_value=int(df['weight'].max()), value=int(df['weight'].median()))
    inp_hp = st.number_input("Horsepower", min_value=int(np.nanmin(df['horsepower'])), max_value=int(np.nanmax(df['horsepower'])), value=int(np.nanmedian(df['horsepower'].dropna())))
    pred = lm.predict(np.array([[inp_weight, inp_hp]]))[0]
    st.write(f"Predicted MPG: {pred:.2f}")

# Download filtered dataset
st.sidebar.markdown("---")
st.sidebar.markdown("Download")
csv = filtered.to_csv(index=False)
st.sidebar.download_button("Download filtered CSV", data=csv, file_name="automobile_filtered.csv", mime="text/csv")

st.markdown("## Notes / Next steps")
st.markdown("- You can extend the predictor (add features, cross-validation, regularization).")
st.markdown("- Add per-origin regression lines, or let the user choose features for the model.")

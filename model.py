# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier

# Evaluating Model
from sklearn.metrics import accuracy_score,classification_report,f1_score, precision_recall_fscore_support,precision_recall_curve,precision_score, confusion_matrix

# Machine Learning Experiment 
import mlflow
import mlflow.sklearn

import joblib

# Define a function to style the DataFrame for better readability
def style_dataframe(df):
    """
    Applies a consistent styling to the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to style.

    Returns:
        pd.io.formats.style.Styler: The styled DataFrame.
    """
    # Use a color palette for header and even/odd rows
    palette = sns.color_palette("coolwarm", n_colors=63)

    return df.style.set_table_styles(
        [
            {
                'selector': 'thead th',
                'props': [
                    ('background-color', palette[0]),  # First color from palette for header
                    ('color', '#FFFFFF'),  # White text for header
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('border', f'1px solid {palette[0]}'),  # Same color as header background for border
                ]
            },
            {
                'selector': 'tbody td',
                'props': [
                    ('background-color', '#FFFFFF'),  # White background for cells
                    ('border', '1px solid #DDDDDD'),  # Light grey border for cells
                    ('color', '#333333'),  # Dark grey text for better readability
                ]
            },
            {
                'selector': 'tbody tr:nth-child(even) td',
                'props': [
                    ('background-color', palette[1])  # Second color from palette for even rows
                ]
            },
            {
                'selector': 'tbody tr:nth-child(odd) td',
                'props': [
                    ('background-color', palette[2])  # Third color from palette for odd rows
                ]
            }
        ]
    ).set_properties(**{'text-align': 'center'}).set_table_attributes('style="width:100%;"').hide(axis='index')

# Read the data from a CSV file (replace 'your_data.csv' with your actual file path)
df = pd.read_csv(r"C:\Users\Moin\Science learning\Data_Projects\Machine Learning\Self_learning\csv\fictional_character_battles.csv")

# Optionally, display the rows using head()
style = style_dataframe(df.head(11))
display(style)  # Use display to render the styled DataFrame

mlflow.set_experiment('battle_prediction')
# Display basic information about the DataFrame
print(df.shape)  # Print the shape of the DataFrame
print(df.describe())  # Print summary statistics of the DataFrame
print(df.info())  # Print information about the DataFrame
print(df.isnull().sum())  # Print the number of missing values in each column
print(df.columns)  # Print the column names

# Unique values in different columns
Strenghth_val = df.Strength.unique()
Weaknesses_val = df.Weaknesses.unique()
Intelligence_val = df.Intelligence.unique()
SpecialAbilities_val = df.SpecialAbilities.unique()
Character_val = df.Character.nunique()
Character_valu = df.Character.unique()

# Print unique values
print(f'Unique Strength In Heros is:', Strenghth_val)
print(f'Unique Weaknesses In Heros is:', Weaknesses_val)
print(f'Unique Intelligence In Heros is:', Intelligence_val)
print(f'Unique SpecialAbilities In Heros is:', SpecialAbilities_val)
print(f'Number Of Unique Heros Names:', Character_val)
print(f'Names Of The Super Heros:', Character_valu)

# Check specific columns
col_check = df[['Character', 'BattleOutcome']]
print(col_check.tail(20))  # Print the last 20 rows

col_check = df[['Strength', 'BattleOutcome']]
print(col_check.iloc[1560:1600, :])  # Print rows from index 1560 to 1600

# Correlation analysis
corr_col = df['Strength'].corr(df['BattleOutcome'])
print(f"Correlation between 'Strength' and 'BattleOutcome': {corr_col:.4f}")

col_check = df[['Speed', 'BattleOutcome']]
print(col_check.iloc[1560:1600, :])  # Print rows from index 1560 to 1600

corr_col2 = df['Speed'].corr(df['BattleOutcome'])
print(f"Correlation between 'Speed' and 'BattleOutcome': {corr_col2:.4f}")

col_check = df[['Intelligence', 'BattleOutcome']]
print(col_check.iloc[1560:1600, :])  # Print rows from index 1560 to 1600

corr_col3 = df['Intelligence'].corr(df['BattleOutcome'])
print(f"Correlation between 'Intelligence' and 'BattleOutcome': {corr_col3:.4f}")

# Separate numeric and character columns
numeric_cols = df.select_dtypes(include=['int64', 'float64'])
charcter_columns = df.select_dtypes(include=['object'])

# Print numeric and character columns
print(numeric_cols)
print(charcter_columns)

# Histograms for numeric columns
df.hist(bins=20, figsize=(10, 5), color=['Teal'])
plt.show()

# Histogram for specific columns
plt.figure(figsize=(15, 4))
plt.hist([df['Character'], df['Speed']], bins=20, alpha=0.7, color=['Teal', 'PaleVioletRed'], label=['Heros', 'Speed'])
plt.xlabel('Hero Name')
plt.ylabel('Speed Of Heros')
plt.title('Distribution of Hero and Speed')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(15, 4))
plt.hist([df['SpecialAbilities'], df['Intelligence']], bins=20, alpha=0.7, color=['Teal', 'PaleVioletRed'], label=['SpecialAbilities', 'Intelligence'])
plt.xlabel('SpecialAbilities Of Heros')
plt.ylabel('Intelligence Of Heros')
plt.title('Distribution of SpecialAbilities and Intelligence')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

plt.figure(figsize=(11, 4))
plt.hist([df['Weaknesses'], df['Strength']], bins=20, alpha=0.7, color=['Teal', 'PaleVioletRed'], label=['Weaknesses', 'Strength'])
plt.xlabel('Weaknesses Of Heros')
plt.ylabel('Frequency')
plt.title('Distribution of Weaknesses and Strength')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()

# Boxplot for Strength vs Weaknesses
font_color = "Teal"  # You can change this to your preference

plt.figure(figsize=(8, 4))
sns.boxplot(
    data=df,
    x="Strength",
    y="Weaknesses",
    palette="husl",
    showmeans=True,  # Display means as markers on the boxes
    notch=True,  # Add notches to the boxes for better variability representation
)
plt.xlabel("Strength", color=font_color)  # X-axis label
plt.ylabel("Character", color=font_color)  # Y-axis label
plt.title("Boxplot of Strength vs Character", color=font_color)  # Title
plt.tick_params(colors=font_color)  # Tick labels
plt.show()

# Boxplot for Strength vs Character
plt.figure(figsize=(8, 4))
sns.boxplot(
    data=df,
    x="Strength",
    y="Character",
    palette="husl",
    showmeans=True,  # Display means as markers on the boxes
    notch=True,  # Add notches to the boxes for better variability representation
)
plt.xlabel("Strength", color=font_color)  # X-axis label
plt.ylabel("Character", color=font_color)  # Y-axis label
plt.title("Boxplot of Strength vs Character", color=font_color)  # Title
plt.tick_params(colors=font_color)  # Tick labels (both axes)
plt.show()

# Pairplot for the dataset
plt.figure(figsize=(10, 8))
sns.pairplot(data=df, hue='BattleOutcome', palette="husl", diag_kind="kde")
plt.show()


df.columns

charcter_columns


# Label Encoding
char_label = LabelEncoder()
universe_label = LabelEncoder()
specialAbilities_label = LabelEncoder()
weakness_label = LabelEncoder()

df['Character'] = char_label.fit_transform(df['Character'])
df['Universe'] = universe_label.fit_transform(df['Universe'])
df['SpecialAbilities'] = specialAbilities_label.fit_transform(df['SpecialAbilities'])
df['Weaknesses'] = weakness_label.fit_transform(df['Weaknesses'])

df.head(5)

df.Character.unique()

# Split Data (Independent and Depended)
X_input_data = df.iloc[:,:-1]
y_output_data = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X_input_data, y_output_data, test_size=0.2, random_state=42)

Ada_model = AdaBoostClassifier()
Ada_model.fit(X_train, y_train)


Ada_model.score(X_train, y_train)*100, Ada_model.score(X_test, y_test)*100


with mlflow.start_run():
    forest_model = RandomForestClassifier(n_estimators=600,
    criterion='entropy',
    max_features='log2',
    n_jobs=1,
    random_state=1,
    verbose=0,
    max_depth=456,
    min_samples_split=2,
    min_samples_leaf=1,
    warm_start=True,)
    forest_model.fit(X_train, y_train)


forest_model.score(X_train, y_train)*100, forest_model.score(X_test, y_test)*100

y_pred = forest_model.predict(X_test)
F1_Score = f1_score(y_pred, y_test)
Accuracy_Score = accuracy_score(y_pred, y_test)
Report = classification_report(y_pred, y_test)
Confusion_Matrix = confusion_matrix(y_pred, y_test)

print(f'F1 Score:',F1_Score)
print(f'Accuracy Score:',Accuracy_Score)
print(Report,f'Classification Report:')
print(Confusion_Matrix)

# precision = Report['weighted avg']['precision']
# recall = Report['weighted avg']['recall']




colour = ['PaleVioletRed','Teal','Pink']
report = classification_report(y_test, y_pred, output_dict=True)

df_report = pd.DataFrame(report).transpose()
# Plot metrics
df_report[['precision', 'recall', 'f1-score']].plot(kind='bar',  color = colour)
plt.title('Classification Report')
plt.ylabel('Metric Value')
plt.xlabel('Class')
plt.xticks(rotation=50)
plt.show()


mlflow.log_metric("F1 Score",F1_Score)
mlflow.log_metric("Accuracy Score",Accuracy_Score)
# mlflow.log_metric("Precsion Score", precision)
# mlflow.log_metric("Recall Score", recall)

Matrix = (Confusion_Matrix.sum(axis=0)[-1] / Confusion_Matrix.sum()) * 100 
mlflow.log_metric("Confusion Matrix", Matrix)

mlflow.set_tag('Algorithm','RandomForestClassifier')
mlflow.set_tag('Data Size', len(df))
mlflow.set_tag('Model Type', 'Classification')
mlflow.set_tag("Random State",42)
mlflow.set_tag('Test Size', '20%')

mlflow.sklearn.log_model(forest_model, 'RandomForestClassifier')

mlflow.log_param('criterian',600)
mlflow.log_param('Max_features','log2')
mlflow.log_param('n_jobs',1)
mlflow.log_param('random_state',1)
mlflow.log_param('verbose',0)
mlflow.log_param('max_depth',456)
mlflow.log_param('min_simples_split',2)
mlflow.log_param('min_samples_leaf',1)
mlflow.log_param('warm_start',True)

# # Aboard the MLflow:
mlflow.end_run()

joblib.dump(char_label,'character_encode')
joblib.dump(universe_label,'universe_encode')
joblib.dump(specialAbilities_label,'Abilities_encode')
joblib.dump(weakness_label,'weaknes_encode')

joblib.dump(forest_model, 'model.joblib')
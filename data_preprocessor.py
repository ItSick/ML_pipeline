import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
    
    def fill_missing(self):
        """Fill numerical columns with median, categorical with mode."""
        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64']:
                print(f"apply median to column {col}")
                self.df[col] = self.df[col].fillna(self.df[col].median())
            else:
                if self.df[col].mode().empty:
                    self.df[col] = self.df[col].fillna("Unknown")
                    print(f"found empty rows in column {col} changed to Unknown")
                else:
                    print(f"apply mode to column {col}")
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        print("fill_missing terminated successfully")

    def detect_outliers_iqr(self):
        """Mark outliers in numerical columns based on IQR."""
        for col in self.df.select_dtypes(include=np.number):
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = ~((self.df[col] >= Q1 - 1.5 * IQR) & (self.df[col] <= Q3 + 1.5 * IQR))
            self.df[col + '_outlier'] = mask.astype(int)
        print("detect_outliers_iqr terminated successfully")

    def encode_categorical(self):
        """One-hot encode categorical variables."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        print("encode_categorical terminated successfully")

    def scale_features(self):
        """Standardize numerical features using StandardScaler."""
        scaler = StandardScaler()
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        print("scale_features terminated successfully")

    def split_to_train_test(self, target_column, test_size=0.2, random_state=42):
        """Split the dataset into train and test sets."""
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def explore_data(self):
        """Run some basic EDA commands."""
        print("First 5 rows:\n", self.df.head())
        print("__________________________________________")
        print("\nShape of the DataFrame:", self.df.shape)
        print("__________________________________________")
        print("\nColumns:\n", self.df.columns.tolist())
        print("__________________________________________")
        print("\nInfo:")
        self.df.info()
        print("__________________________________________")
        print("\nData Types:\n", self.df.dtypes)
        print("__________________________________________")
        print("\nUnique values per column:\n", self.df.nunique())
        print("__________________________________________")

    def get_clean_data(self):
        """Return the processed DataFrame."""
        return self.df

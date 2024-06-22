loan_path = "./data/raw/loan.csv"
payment_path = "./data/raw/payment.csv"
underwriting_path = "./data/raw/clarity_underwriting_variables.csv"

# Load data
loan_df, payment_df, underwriting_df = load_data(loan_path, payment_path, underwriting_path)

# Preprocess data
df = preprocess_data(loan_df, underwriting_df)

# Classify loans
df = classify_loans(df)

num_feats, freq_feats, target_feats, predictor = define_features()

selected_features = list(set(num_feats + freq_feats + target_feats + predictor))

# Split data into train and test sets
df = df[selected_features]
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = train_df[selected_features]

# Initialize the encoder
encoder = DataEncoder(num_feats, target_feats, freq_feats)

# Encode the training data
X_train = encoder.fit_transform(train_df, predictor)
y_train = train_df[predictor]

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_test = encoder.transform(test_df)
y_test = test_df[predictor]
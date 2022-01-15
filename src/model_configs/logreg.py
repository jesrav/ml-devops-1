
categorical_feature_pipeline = Pipeline([
        ("column_selector", ColumnTransformer(transformers=[('selector', 'passthrough', CATEGORICAL_COLS)], remainder="drop")),
        ('imputation', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('encoding', OneHotEncoder()),
])

numerical_feature_pipeline = Pipeline conda ([
        ("column_selector", ColumnTransformer(transformers=[('selector', 'passthrough', NUMERICAL_COLS)], remainder="drop")),
        ('imputation', SimpleImputer(missing_values=np.nan, strategy='mean'))
])


feature_union = FeatureUnion([
    ("categorical", categorical_feature_pipeline),
    ("non_categorical", numerical_feature_pipeline)
])

ml_pipeline = PMMLPipeline([
    ("feature_preprocess", feature_union),
    ("classifier", RandomForestClassifier())
])


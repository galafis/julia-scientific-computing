module MachineLearning

using Statistics
using LinearAlgebra
using Random
using Distributions
using MLJ
using DataFrames
using StatsBase

export split_data, normalize_data, standardize_data
export linear_regression, logistic_regression, decision_tree
export random_forest, gradient_boosting, neural_network
export kmeans_clustering, hierarchical_clustering, dbscan_clustering
export pca, lda, t_sne
export cross_validation, grid_search, evaluate_model
export confusion_matrix, roc_curve, precision_recall_curve

"""
    split_data(X, y, train_ratio=0.8; shuffle=true, stratify=false)

Split data into training and testing sets.

# Arguments
- `X`: Feature matrix
- `y`: Target vector
- `train_ratio`: Ratio of training data
- `shuffle`: Whether to shuffle the data
- `stratify`: Whether to stratify the split by target classes

# Returns
- Training and testing data (X_train, y_train, X_test, y_test)
"""
function split_data(X, y, train_ratio=0.8; shuffle=true, stratify=false)
    n = size(X, 1)
    
    if shuffle
        # Shuffle indices
        indices = shuffle(1:n)
        X = X[indices, :]
        y = y[indices]
    end
    
    if stratify && length(unique(y)) > 1
        # Stratified split
        classes = unique(y)
        train_indices = Int[]
        
        for class in classes
            class_indices = findall(y .== class)
            n_class = length(class_indices)
            n_train = round(Int, train_ratio * n_class)
            
            append!(train_indices, class_indices[1:n_train])
        end
        
        test_indices = setdiff(1:n, train_indices)
    else
        # Random split
        n_train = round(Int, train_ratio * n)
        train_indices = 1:n_train
        test_indices = (n_train+1):n
    end
    
    X_train = X[train_indices, :]
    y_train = y[train_indices]
    X_test = X[test_indices, :]
    y_test = y[test_indices]
    
    return X_train, y_train, X_test, y_test
end

"""
    normalize_data(X; dims=1)

Normalize data to [0, 1] range.

# Arguments
- `X`: Data matrix
- `dims`: Dimension along which to normalize (1 for columns, 2 for rows)

# Returns
- Normalized data and normalization parameters
"""
function normalize_data(X; dims=1)
    X_min = minimum(X, dims=dims)
    X_max = maximum(X, dims=dims)
    
    # Avoid division by zero
    X_range = X_max - X_min
    X_range[X_range .== 0] .= 1.0
    
    X_norm = (X .- X_min) ./ X_range
    
    return X_norm, (X_min, X_range)
end

"""
    standardize_data(X; dims=1)

Standardize data to zero mean and unit variance.

# Arguments
- `X`: Data matrix
- `dims`: Dimension along which to standardize (1 for columns, 2 for rows)

# Returns
- Standardized data and standardization parameters
"""
function standardize_data(X; dims=1)
    X_mean = mean(X, dims=dims)
    X_std = std(X, dims=dims)
    
    # Avoid division by zero
    X_std[X_std .== 0] .= 1.0
    
    X_stand = (X .- X_mean) ./ X_std
    
    return X_stand, (X_mean, X_std)
end

"""
    linear_regression(X, y; method=:ols)

Fit a linear regression model.

# Arguments
- `X`: Feature matrix
- `y`: Target vector
- `method`: Method to use (:ols, :ridge, :lasso)

# Returns
- Fitted model
"""
function linear_regression(X, y; method=:ols)
    if method == :ols
        # Ordinary least squares
        LinearRegressor = @load LinearRegressor pkg=MLJLinearModels
        model = LinearRegressor()
    elseif method == :ridge
        # Ridge regression
        RidgeRegressor = @load RidgeRegressor pkg=MLJLinearModels
        model = RidgeRegressor()
    elseif method == :lasso
        # Lasso regression
        LassoRegressor = @load LassoRegressor pkg=MLJLinearModels
        model = LassoRegressor()
    else
        error("Unknown method: $method")
    end
    
    # Create machine
    mach = machine(model, X, y)
    
    # Fit model
    fit!(mach)
    
    return mach
end

"""
    logistic_regression(X, y; penalty=:none, C=1.0)

Fit a logistic regression model.

# Arguments
- `X`: Feature matrix
- `y`: Target vector
- `penalty`: Penalty type (:none, :l1, :l2)
- `C`: Inverse of regularization strength

# Returns
- Fitted model
"""
function logistic_regression(X, y; penalty=:none, C=1.0)
    if penalty == :none
        # No regularization
        LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
        model = LogisticClassifier()
    elseif penalty == :l1
        # L1 regularization
        LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
        model = LogisticClassifier(penalty=:l1, lambda=1/C)
    elseif penalty == :l2
        # L2 regularization
        LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels
        model = LogisticClassifier(penalty=:l2, lambda=1/C)
    else
        error("Unknown penalty: $penalty")
    end
    
    # Create machine
    mach = machine(model, X, y)
    
    # Fit model
    fit!(mach)
    
    return mach
end

"""
    decision_tree(X, y; max_depth=nothing, min_samples_split=2, criterion=:gini)

Fit a decision tree model.

# Arguments
- `X`: Feature matrix
- `y`: Target vector
- `max_depth`: Maximum depth of the tree
- `min_samples_split`: Minimum number of samples required to split a node
- `criterion`: Split criterion (:gini or :entropy for classification, :mse for regression)

# Returns
- Fitted model
"""
function decision_tree(X, y; max_depth=nothing, min_samples_split=2, criterion=:gini)
    if eltype(y) <: Number && length(unique(y)) > 10
        # Regression
        DecisionTreeRegressor = @load DecisionTreeRegressor pkg=DecisionTree
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
    else
        # Classification
        DecisionTreeClassifier = @load DecisionTreeClassifier pkg=DecisionTree
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion
        )
    end
    
    # Create machine
    mach = machine(model, X, y)
    
    # Fit model
    fit!(mach)
    
    return mach
end

"""
    random_forest(X, y; n_estimators=100, max_depth=nothing, min_samples_split=2, criterion=:gini)

Fit a random forest model.

# Arguments
- `X`: Feature matrix
- `y`: Target vector
- `n_estimators`: Number of trees in the forest
- `max_depth`: Maximum depth of the trees
- `min_samples_split`: Minimum number of samples required to split a node
- `criterion`: Split criterion (:gini or :entropy for classification, :mse for regression)

# Returns
- Fitted model
"""
function random_forest(X, y; n_estimators=100, max_depth=nothing, min_samples_split=2, criterion=:gini)
    if eltype(y) <: Number && length(unique(y)) > 10
        # Regression
        RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree
        model = RandomForestRegressor(
            n_trees=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
    else
        # Classification
        RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree
        model = RandomForestClassifier(
            n_trees=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            criterion=criterion
        )
    end
    
    # Create machine
    mach = machine(model, X, y)
    
    # Fit model
    fit!(mach)
    
    return mach
end

"""
    gradient_boosting(X, y; n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0)

Fit a gradient boosting model.

# Arguments
- `X`: Feature matrix
- `y`: Target vector
- `n_estimators`: Number of boosting stages
- `learning_rate`: Learning rate
- `max_depth`: Maximum depth of the trees
- `subsample`: Fraction of samples to use for fitting the base learners

# Returns
- Fitted model
"""
function gradient_boosting(X, y; n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0)
    if eltype(y) <: Number && length(unique(y)) > 10
        # Regression
        GradientBoostingRegressor = @load GradientBoostingRegressor pkg=LightGBM
        model = GradientBoostingRegressor(
            num_iterations=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            feature_fraction=subsample
        )
    else
        # Classification
        GradientBoostingClassifier = @load GradientBoostingClassifier pkg=LightGBM
        model = GradientBoostingClassifier(
            num_iterations=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            feature_fraction=subsample
        )
    end
    
    # Create machine
    mach = machine(model, X, y)
    
    # Fit model
    fit!(mach)
    
    return mach
end

"""
    neural_network(X, y; hidden_layers=[100], activation=:relu, learning_rate=0.01, epochs=100, batch_size=32)

Fit a neural network model.

# Arguments
- `X`: Feature matrix
- `y`: Target vector
- `hidden_layers`: List of hidden layer sizes
- `activation`: Activation function (:relu, :sigmoid, :tanh)
- `learning_rate`: Learning rate
- `epochs`: Number of epochs
- `batch_size`: Batch size

# Returns
- Fitted model
"""
function neural_network(X, y; hidden_layers=[100], activation=:relu, learning_rate=0.01, epochs=100, batch_size=32)
    if eltype(y) <: Number && length(unique(y)) > 10
        # Regression
        NeuralNetworkRegressor = @load NeuralNetworkRegressor pkg=MLJFlux
        model = NeuralNetworkRegressor(
            builder=MLJFlux.MLP(hidden_layers, activation),
            optimiser=Flux.ADAM(learning_rate),
            epochs=epochs,
            batch_size=batch_size
        )
    else
        # Classification
        NeuralNetworkClassifier = @load NeuralNetworkClassifier pkg=MLJFlux
        model = NeuralNetworkClassifier(
            builder=MLJFlux.MLP(hidden_layers, activation),
            optimiser=Flux.ADAM(learning_rate),
            epochs=epochs,
            batch_size=batch_size
        )
    end
    
    # Create machine
    mach = machine(model, X, y)
    
    # Fit model
    fit!(mach)
    
    return mach
end

"""
    kmeans_clustering(X; n_clusters=8, max_iter=300, tol=1e-4)

Perform K-means clustering.

# Arguments
- `X`: Data matrix
- `n_clusters`: Number of clusters
- `max_iter`: Maximum number of iterations
- `tol`: Tolerance for convergence

# Returns
- Cluster assignments and centroids
"""
function kmeans_clustering(X; n_clusters=8, max_iter=300, tol=1e-4)
    KMeans = @load KMeans pkg=Clustering
    model = KMeans(k=n_clusters, tol=tol, max_iter=max_iter)
    
    # Create machine
    mach = machine(model, X)
    
    # Fit model
    fit!(mach)
    
    # Get cluster assignments and centroids
    clusters = predict(mach)
    centroids = mach.fitresult.centers
    
    return clusters, centroids
end

"""
    hierarchical_clustering(X; n_clusters=8, linkage=:ward)

Perform hierarchical clustering.

# Arguments
- `X`: Data matrix
- `n_clusters`: Number of clusters
- `linkage`: Linkage method (:ward, :single, :complete, :average)

# Returns
- Cluster assignments and dendrogram
"""
function hierarchical_clustering(X; n_clusters=8, linkage=:ward)
    # Compute distance matrix
    dist_matrix = pairwise(Euclidean(), X', X')
    
    # Perform hierarchical clustering
    hc = hclust(dist_matrix, linkage=linkage)
    
    # Cut the dendrogram to get clusters
    clusters = cutree(hc, k=n_clusters)
    
    return clusters, hc
end

"""
    dbscan_clustering(X; eps=0.5, min_samples=5)

Perform DBSCAN clustering.

# Arguments
- `X`: Data matrix
- `eps`: Maximum distance between two samples for them to be considered as in the same neighborhood
- `min_samples`: Minimum number of samples in a neighborhood for a point to be considered as a core point

# Returns
- Cluster assignments
"""
function dbscan_clustering(X; eps=0.5, min_samples=5)
    DBSCAN = @load DBSCAN pkg=Clustering
    model = DBSCAN(eps=eps, min_neighbors=min_samples)
    
    # Create machine
    mach = machine(model, X)
    
    # Fit model
    fit!(mach)
    
    # Get cluster assignments
    clusters = predict(mach)
    
    return clusters
end

"""
    pca(X; n_components=nothing, whiten=false)

Perform principal component analysis.

# Arguments
- `X`: Data matrix
- `n_components`: Number of components to keep
- `whiten`: Whether to whiten the data

# Returns
- Transformed data and PCA model
"""
function pca(X; n_components=nothing, whiten=false)
    PCA = @load PCA pkg=MultivariateStats
    
    if isnothing(n_components)
        n_components = min(size(X)...)
    end
    
    model = PCA(n_components=n_components, whiten=whiten)
    
    # Create machine
    mach = machine(model, X)
    
    # Fit model
    fit!(mach)
    
    # Transform data
    X_transformed = transform(mach, X)
    
    return X_transformed, mach
end

"""
    lda(X, y; n_components=nothing)

Perform linear discriminant analysis.

# Arguments
- `X`: Data matrix
- `y`: Target vector
- `n_components`: Number of components to keep

# Returns
- Transformed data and LDA model
"""
function lda(X, y; n_components=nothing)
    LDA = @load LDA pkg=MultivariateStats
    
    if isnothing(n_components)
        n_components = min(length(unique(y)) - 1, size(X, 2))
    end
    
    model = LDA(n_components=n_components)
    
    # Create machine
    mach = machine(model, X, y)
    
    # Fit model
    fit!(mach)
    
    # Transform data
    X_transformed = transform(mach, X)
    
    return X_transformed, mach
end

"""
    t_sne(X; n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000)

Perform t-SNE dimensionality reduction.

# Arguments
- `X`: Data matrix
- `n_components`: Number of components to keep
- `perplexity`: Perplexity parameter
- `learning_rate`: Learning rate
- `n_iter`: Number of iterations

# Returns
- Transformed data
"""
function t_sne(X; n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000)
    TSNE = @load TSNE pkg=TSne
    model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter
    )
    
    # Create machine
    mach = machine(model, X)
    
    # Fit model
    fit!(mach)
    
    # Transform data
    X_transformed = transform(mach, X)
    
    return X_transformed
end

"""
    cross_validation(model, X, y; n_folds=5, shuffle=true, stratify=false)

Perform cross-validation.

# Arguments
- `model`: Model to evaluate
- `X`: Feature matrix
- `y`: Target vector
- `n_folds`: Number of folds
- `shuffle`: Whether to shuffle the data
- `stratify`: Whether to stratify the folds by target classes

# Returns
- Cross-validation scores
"""
function cross_validation(model, X, y; n_folds=5, shuffle=true, stratify=false)
    # Create cross-validation iterator
    if stratify && length(unique(y)) > 1
        cv = StratifiedKFold(n_folds, shuffle=shuffle)
    else
        cv = KFold(n_folds, shuffle=shuffle)
    end
    
    # Initialize scores
    scores = []
    
    # Perform cross-validation
    for (train_indices, test_indices) in cv(X, y)
        X_train = X[train_indices, :]
        y_train = y[train_indices]
        X_test = X[test_indices, :]
        y_test = y[test_indices]
        
        # Create and fit machine
        mach = machine(model, X_train, y_train)
        fit!(mach)
        
        # Evaluate model
        y_pred = predict(mach, X_test)
        
        # Compute score
        if eltype(y) <: Number && length(unique(y)) > 10
            # Regression
            score = r2_score(y_test, y_pred)
        else
            # Classification
            score = accuracy(y_test, y_pred)
        end
        
        push!(scores, score)
    end
    
    return scores
end

"""
    grid_search(model, X, y, param_grid; n_folds=5, shuffle=true, stratify=false)

Perform grid search for hyperparameter tuning.

# Arguments
- `model`: Model to tune
- `X`: Feature matrix
- `y`: Target vector
- `param_grid`: Dictionary of parameter names and values
- `n_folds`: Number of folds for cross-validation
- `shuffle`: Whether to shuffle the data
- `stratify`: Whether to stratify the folds by target classes

# Returns
- Best parameters and best score
"""
function grid_search(model, X, y, param_grid; n_folds=5, shuffle=true, stratify=false)
    # Generate all parameter combinations
    param_names = keys(param_grid)
    param_values = values(param_grid)
    param_combinations = Iterators.product(param_values...)
    
    # Initialize best parameters and score
    best_params = nothing
    best_score = -Inf
    
    # Iterate over parameter combinations
    for params in param_combinations
        # Set model parameters
        for (name, value) in zip(param_names, params)
            setproperty!(model, name, value)
        end
        
        # Perform cross-validation
        scores = cross_validation(model, X, y, n_folds=n_folds, shuffle=shuffle, stratify=stratify)
        
        # Compute mean score
        mean_score = mean(scores)
        
        # Update best parameters and score
        if mean_score > best_score
            best_score = mean_score
            best_params = Dict(name => value for (name, value) in zip(param_names, params))
        end
    end
    
    return best_params, best_score
end

"""
    evaluate_model(model, X_train, y_train, X_test, y_test)

Evaluate a model on training and testing data.

# Arguments
- `model`: Model to evaluate
- `X_train`: Training feature matrix
- `y_train`: Training target vector
- `X_test`: Testing feature matrix
- `y_test`: Testing target vector

# Returns
- Dictionary with evaluation metrics
"""
function evaluate_model(model, X_train, y_train, X_test, y_test)
    # Create and fit machine
    mach = machine(model, X_train, y_train)
    fit!(mach)
    
    # Make predictions
    y_train_pred = predict(mach, X_train)
    y_test_pred = predict(mach, X_test)
    
    # Initialize metrics
    metrics = Dict()
    
    if eltype(y_train) <: Number && length(unique(y_train)) > 10
        # Regression metrics
        metrics["train_r2"] = r2_score(y_train, y_train_pred)
        metrics["test_r2"] = r2_score(y_test, y_test_pred)
        
        metrics["train_mse"] = mean_squared_error(y_train, y_train_pred)
        metrics["test_mse"] = mean_squared_error(y_test, y_test_pred)
        
        metrics["train_mae"] = mean_absolute_error(y_train, y_train_pred)
        metrics["test_mae"] = mean_absolute_error(y_test, y_test_pred)
    else
        # Classification metrics
        metrics["train_accuracy"] = accuracy(y_train, y_train_pred)
        metrics["test_accuracy"] = accuracy(y_test, y_test_pred)
        
        if length(unique(y_train)) == 2
            # Binary classification
            metrics["train_precision"] = precision(y_train, y_train_pred)
            metrics["test_precision"] = precision(y_test, y_test_pred)
            
            metrics["train_recall"] = recall(y_train, y_train_pred)
            metrics["test_recall"] = recall(y_test, y_test_pred)
            
            metrics["train_f1"] = f1_score(y_train, y_train_pred)
            metrics["test_f1"] = f1_score(y_test, y_test_pred)
            
            # ROC and AUC
            train_fpr, train_tpr, _ = roc_curve(y_train, y_train_pred)
            test_fpr, test_tpr, _ = roc_curve(y_test, y_test_pred)
            
            metrics["train_auc"] = auc(train_fpr, train_tpr)
            metrics["test_auc"] = auc(test_fpr, test_tpr)
        else
            # Multi-class classification
            metrics["train_precision_macro"] = precision(y_train, y_train_pred, average=:macro)
            metrics["test_precision_macro"] = precision(y_test, y_test_pred, average=:macro)
            
            metrics["train_recall_macro"] = recall(y_train, y_train_pred, average=:macro)
            metrics["test_recall_macro"] = recall(y_test, y_test_pred, average=:macro)
            
            metrics["train_f1_macro"] = f1_score(y_train, y_train_pred, average=:macro)
            metrics["test_f1_macro"] = f1_score(y_test, y_test_pred, average=:macro)
        end
    end
    
    return metrics
end

"""
    confusion_matrix(y_true, y_pred; normalize=false)

Compute confusion matrix.

# Arguments
- `y_true`: True labels
- `y_pred`: Predicted labels
- `normalize`: Whether to normalize the confusion matrix

# Returns
- Confusion matrix
"""
function confusion_matrix(y_true, y_pred; normalize=false)
    # Get unique classes
    classes = sort(unique(vcat(y_true, y_pred)))
    n_classes = length(classes)
    
    # Initialize confusion matrix
    cm = zeros(Int, n_classes, n_classes)
    
    # Fill confusion matrix
    for (i, true_class) in enumerate(classes)
        for (j, pred_class) in enumerate(classes)
            cm[i, j] = sum((y_true .== true_class) .& (y_pred .== pred_class))
        end
    end
    
    # Normalize if requested
    if normalize
        cm = cm ./ sum(cm, dims=2)
    end
    
    return cm, classes
end

"""
    roc_curve(y_true, y_score; pos_label=nothing)

Compute Receiver Operating Characteristic (ROC) curve.

# Arguments
- `y_true`: True binary labels
- `y_score`: Target scores (probabilities or decision function outputs)
- `pos_label`: Label of the positive class

# Returns
- False positive rate, true positive rate, and thresholds
"""
function roc_curve(y_true, y_score; pos_label=nothing)
    # Determine positive class
    if isnothing(pos_label)
        pos_label = maximum(y_true)
    end
    
    # Convert to binary problem
    y_true_bin = y_true .== pos_label
    
    # Sort scores and corresponding truth values
    desc_score_indices = sortperm(y_score, rev=true)
    y_score = y_score[desc_score_indices]
    y_true_bin = y_true_bin[desc_score_indices]
    
    # Count positive and negative samples
    n_pos = sum(y_true_bin)
    n_neg = length(y_true_bin) - n_pos
    
    # Compute true positive and false positive rates
    tps = cumsum(y_true_bin)
    fps = cumsum(.!y_true_bin)
    
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Add (0, 0) and (1, 1) points
    tpr = vcat(0, tpr, 1)
    fpr = vcat(0, fpr, 1)
    
    # Compute thresholds
    thresholds = vcat(maximum(y_score) + 1, y_score, minimum(y_score) - 1)
    
    return fpr, tpr, thresholds
end

"""
    precision_recall_curve(y_true, y_score; pos_label=nothing)

Compute Precision-Recall curve.

# Arguments
- `y_true`: True binary labels
- `y_score`: Target scores (probabilities or decision function outputs)
- `pos_label`: Label of the positive class

# Returns
- Precision, recall, and thresholds
"""
function precision_recall_curve(y_true, y_score; pos_label=nothing)
    # Determine positive class
    if isnothing(pos_label)
        pos_label = maximum(y_true)
    end
    
    # Convert to binary problem
    y_true_bin = y_true .== pos_label
    
    # Sort scores and corresponding truth values
    desc_score_indices = sortperm(y_score, rev=true)
    y_score = y_score[desc_score_indices]
    y_true_bin = y_true_bin[desc_score_indices]
    
    # Compute precision and recall
    tps = cumsum(y_true_bin)
    fps = cumsum(.!y_true_bin)
    
    precision = tps ./ (tps + fps)
    recall = tps / sum(y_true_bin)
    
    # Add (0, 1) point
    precision = vcat(1, precision)
    recall = vcat(0, recall)
    
    # Compute thresholds
    thresholds = vcat(maximum(y_score) + 1, y_score)
    
    return precision, recall, thresholds
end

end # module


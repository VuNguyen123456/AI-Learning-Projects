import pandas as pd
import numpy as np
import random
from collections import Counter

# # Set seed for reproducibility
# random.seed(42)
# np.random.seed(42)

# Define possible values for each attribute
attributes = {
    'Alternate': [True, False],
    'Bar': [True, False],
    'Fri/Sat': [True, False],
    'Hungry': [True, False],
    'Patrons': ['None', 'Some', 'Full'],
    'Price': ['$', '$$', '$$$'],
    'Raining': [True, False],
    'Reservation': [True, False],
    'Type': ['French', 'Italian', 'Thai', 'Burger'],
    'WaitEstimate': ['0-10', '10-30', '30-60', '>60']
}

def determine_will_wait(example):
    """
    Complete decision tree logic for WillWait based on your tree
    """
    
    # Level 1: PATRONS
    if example['Patrons'] == 'None':
        return False
    
    elif example['Patrons'] == 'Some':
        return True
    
    elif example['Patrons'] == 'Full':
        # Level 2: WAITESTIMATE
        
        if example['WaitEstimate'] == '>60':
            return False
        
        elif example['WaitEstimate'] == '0-10':
            return True
        
        elif example['WaitEstimate'] == '30-60':
            # Level 3: ALTERNATE (for 30-60 min wait)
            
            if example['Alternate'] == False:
                # Level 4: RESERVATION
                
                if example['Reservation'] == True:
                    return True
                else:  # Reservation == False
                    # Level 5: BAR
                    if example['Bar'] == True:
                        return True
                    else:
                        return False
            
            else:  # Alternate == True
                # Level 4: FRI/SAT
                if example['Fri/Sat'] == True:
                    return True
                else:
                    return False
        
        elif example['WaitEstimate'] == '10-30':
            # Level 3: HUNGRY (for 10-30 min wait)
            
            if example['Hungry'] == False:
                return True
            
            else:  # Hungry == True
                # Level 4: ALTERNATE (different context)
                
                if example['Alternate'] == False:
                    return True
                
                else:  # Alternate == True
                    # Level 5: RAINING
                    if example['Raining'] == True:
                        return True
                    else:
                        return False
    
    return False  # Default fallback


def generate_restaurant_data(n_samples=100, balance=True, seed=42):
    """
    Generate n_samples examples with balanced WillWait outcomes
    
    Args:
        n_samples: Total number of examples to generate (default 100)
        balance: If True, ensure 50/50 split of Yes/No labels
    
    Returns:
        DataFrame with all examples
    """
    random.seed(seed)
    np.random.seed(seed)

    data = []
    yes_count = 0
    no_count = 0
    target_per_class = n_samples // 2  # 50 each
    
    max_attempts = n_samples * 100  # Prevent infinite loop
    attempts = 0
    
    print(f"Generating {n_samples} examples...")
    print(f"Target: {target_per_class} 'Yes' and {target_per_class} 'No' examples\n")
    
    while len(data) < n_samples and attempts < max_attempts:
        attempts += 1
        
        # Randomly generate an example
        example = {
            'Alternate': random.choice(attributes['Alternate']),
            'Bar': random.choice(attributes['Bar']),
            'Fri/Sat': random.choice(attributes['Fri/Sat']),
            'Hungry': random.choice(attributes['Hungry']),
            'Patrons': random.choice(attributes['Patrons']),
            'Price': random.choice(attributes['Price']),
            'Raining': random.choice(attributes['Raining']),
            'Reservation': random.choice(attributes['Reservation']),
            'Type': random.choice(attributes['Type']),
            'WaitEstimate': random.choice(attributes['WaitEstimate'])
        }
        
        # Determine label based on decision tree
        will_wait = determine_will_wait(example)
        
        # Balance the dataset
        if balance:
            if will_wait and yes_count < target_per_class:
                example['WillWait'] = True
                data.append(example)
                yes_count += 1
                if yes_count % 10 == 0:
                    print(f"Generated {yes_count} 'Yes' examples...")
            elif not will_wait and no_count < target_per_class:
                example['WillWait'] = False
                data.append(example)
                no_count += 1
                if no_count % 10 == 0:
                    print(f"Generated {no_count} 'No' examples...")
        else:
            example['WillWait'] = will_wait
            data.append(example)
    
    # after the loop ends
    if balance and (yes_count != target_per_class or no_count != target_per_class):
        print("Warning: desired balance not achieved. yes:", yes_count, "no:", no_count)
        # Optionally: raise ValueError("Could not generate balanced dataset") or return what we have

    print(f"\nGeneration complete!")
    print(f"Total attempts: {attempts}")
    print(f"Final dataset size: {len(data)}")
    
    return pd.DataFrame(data)


def create_train_test_split(df, train_ratio=0.2, random_state=42):
    # Separate by class
    yes_examples = df[df['WillWait'] == True]
    no_examples = df[df['WillWait'] == False]

    # Calculate split sizes
    n_train_yes = int(len(yes_examples) * train_ratio)
    n_train_no = int(len(no_examples) * train_ratio)

    # Random sampling of indices
    train_yes_idx = yes_examples.sample(n=n_train_yes, random_state=random_state).index
    train_no_idx = no_examples.sample(n=n_train_no, random_state=random_state).index

    train_indices = train_yes_idx.union(train_no_idx)
    train_df = df.loc[train_indices].sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = df.drop(train_indices).reset_index(drop=True)

    return train_df, test_df



def display_data_statistics(df):
    """
    Display comprehensive statistics about the dataset
    """
    print("="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    print(f"\nTotal examples: {len(df)}")
    print(f"\nWillWait Distribution:")
    print(df['WillWait'].value_counts().to_string())
    print(f"\nBalance: {df['WillWait'].value_counts(normalize=True).to_dict()}")
    
    print("\n" + "="*70)
    print("FEATURE DISTRIBUTIONS")
    print("="*70)
    
    for col in df.columns:
        if col != 'WillWait':
            print(f"\n{col}:")
            counts = df[col].value_counts()
            for val, count in counts.items():
                pct = (count / len(df)) * 100
                print(f"  {val}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*70)
    print("WILLWAIT BREAKDOWN BY KEY FEATURES")
    print("="*70)
    
    for feature in ['Patrons', 'WaitEstimate', 'Hungry', 'Alternate', 'Reservation']:
        print(f"\n{feature}:")
        crosstab = pd.crosstab(df[feature], df['WillWait'], margins=True)
        print(crosstab.to_string())

# ============================================================================
# Decision Tree Model
# ============================================================================
# -------------------------
# Step 1: Plurality function
# -------------------------
3# This will return out best guess when we can't perfectly classify examples => it'll reutnr the majority vote
def plurality_value(examples, label_name='WillWait'):
    """
    Returns the most common class in the examples
    """
    labels = [ex[label_name] for ex in examples] # Extract example wheather they will wait or not in this array
    counter = Counter(labels) # Count number of true and false
    most_common_label, count = counter.most_common(1)[0] # Return the most common label
    return most_common_label

# -------------------------
# Step 2: Select "most important attribute"
# -------------------------
# For simplicity in this example, we'll pick the attribute that **splits the examples most evenly**
# A better approach would be information gain or Gini index, but this suffices to illustrate recursion

# We need to pick which attribute to split on. A good split separates the classes.
# A good split is one where each group is mostly one class. (pure)
# A bad split is one where each group still has mixed classes. (impure)

# GOOD SPLIT
    # Each branch gives a clear answer:
    # “If Patrons = None → always NO”
    # “If Patrons = Full → always YES”
        # That means the attribute is very good at predicting the answer.

# BAD SPLIT
    # The branches are still confusing:
    # “If Bar = True → sometimes YES, sometimes NO”
    # “If Bar = False → sometimes YES, sometimes NO”
        # This attribute does not help the tree make a decision.

# examples: list of examples (dictionaries)
# attributes: list of attribute to consider for splitting
def importance_attribute(examples, attributes, label_name='WillWait'):
    """
    Pick the best attribute to split on.
    We choose the attribute that produces the "cleanest" groups.
    Clean = all labels the same
    Messy = mixed labels
    Lower score = cleaner split
    """
    best_attr = None
    best_score = float("inf")  # start with something very large

    # Try each possible attribute
    for attr in attributes:
        # Find all possible values this attribute can take in our examples
        values = set(ex[attr] for ex in examples)

        score = 0  # lower = better

        # Look at what happens if we split by this attribute
        for val in values:
            # All examples where this attribute equals this value
            subset = [ex for ex in examples if ex[attr] == val]

            # Collect the labels (True/False)
            labels = [ex[label_name] for ex in subset]

            # len(set(labels)) tells us how many different labels there are:
            #    1 → all labels the same (clean group)
            #    2 → mixed labels (messy group)
            score += len(set(labels))

        # We want the smallest score (cleanest split)
        if score < best_score:
            best_score = score
            best_attr = attr

    return best_attr


# -------------------------
# Step 3: Recursive learnDecisionTree
# -------------------------
def learn_decision_tree(examples, attributes, parent_examples=[], label_name='WillWait'):
    # examples: current set of data to classify
    # attributes: list of attributes to consider for splitting
    # parent_examples: examples from the parent node (fallback for empty examples)
    # label_name: name of the label field
    """
    Build a decision tree using the recursive algorithm.
    The tree is represented as nested dictionaries:
    {'Attribute': {value1: subtree1, value2: subtree2, ...}}
    Leaf nodes are the label (True/False)
    """
    # -------------------------
    # Base case 1: no examples => return plurality from parent examples => Decision made based on parent
    # -------------------------
    if not examples:
        # Return majority label from parent examples
        return plurality_value(parent_examples)
    
    # -------------------------
    # Base case 2: all examples have same classification => Leaf node => Decision made
    # -------------------------
    labels = [ex[label_name] for ex in examples]
    if all(l == labels[0] for l in labels):
        return labels[0]
    
    # -------------------------
    # Base case 3: no attributes left => return plurality from examples => Decision made based on majority vote right here
    # -------------------------
    if not attributes:
        return plurality_value(examples)
    
    # -------------------------
    # Otherwise: select best attribute to split on
    # -------------------------
    
    # -------------------------
    # STEP 4: Choose the best attribute to split on
    # -------------------------
    best_attr = importance_attribute(examples, attributes)

    # We will create a decision tree node shaped like:
    # { best_attr : { value1: subtree1, value2: subtree2, ... } }
    #
    # Start with empty branches:
    tree = {best_attr: {}}

    # -------------------------
    # STEP 5: Get all unique values of this attribute found in examples
    # e.g. if best_attr = "Patrons" → {"None", "Some", "Full"}
    # -------------------------
    values = set(ex[best_attr] for ex in examples)

    # -------------------------
    # STEP 6: For each value, create a branch in the tree
    # -------------------------
    for val in values:

        # -----------------------------------------------
        # Get all examples where best_attr == this value
        # e.g. if val = "None", take only rows where Patrons="None"
        # This creates a smaller dataset (subproblem)
        # -----------------------------------------------
        new_examples = [ex for ex in examples if ex[best_attr] == val]

        # -----------------------------------------------
        # Remove the attribute we just used (best_attr)
        # because once we split on an attribute, we never split on it again
        # -----------------------------------------------
        remaining_attrs = [a for a in attributes if a != best_attr]

        # -----------------------------------------------
        # Recursively build a subtree on the smaller dataset
        # This handles the three base cases OR continues splitting deeper
        # -----------------------------------------------
        subtree = learn_decision_tree(
            new_examples,          # smaller set of examples
            remaining_attrs,        # attributes left to test
            examples,               # parent examples (for fallback)
        )

        # -----------------------------------------------
        # Attach the subtree to this branch of the current tree
        # Example:
        # tree["Patrons"]["None"] = False
        # tree["Patrons"]["Full"]  = { ... deeper tree ... }
        # -----------------------------------------------
        tree[best_attr][val] = subtree

    # At this point, all branches for this attribute are handled
    return tree

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

# -------------------------
# Helper: classify a single example using the nested-dict tree
# -------------------------
def classify(tree, example, default=None):
    """
    Traverse the tree to predict the label for `example`.
    - tree: nested dict or a label (leaf)
    - example: dict with features
    - default: fallback label if attribute/value not found
    """
    # If tree is a leaf (True/False or string), return it
    if not isinstance(tree, dict):
        return tree

    # tree is like { attribute_name: {value1: subtree1, value2: subtree2, ...} }
    attr = next(iter(tree))             # root attribute at this node
    branches = tree[attr]               # dict mapping attribute values -> subtree

    val = example.get(attr, None)       # the example's value for this attribute
    # If the value exists as a branch, follow it
    if val in branches:
        return classify(branches[val], example, default)
    else:
        # Unknown value (not seen during training) -> fallback to default
        # If default is None, pick plurality of training examples stored in subtree? but here we use default param
        return default

# -------------------------
# Decision Tree Tester
# -------------------------
def test_decision_tree(train_df, test_df, label_name="WillWait"):
    """
    Train a decision tree on train_df and evaluate on test_df.
    Returns the tree, default label, train accuracy, and test accuracy.
    """
    # Convert DataFrame to list-of-dicts
    train_examples = train_df.to_dict(orient="records")
    test_examples  = test_df.to_dict(orient="records")

    # Attributes to consider for splitting
    attributes = [col for col in train_df.columns if col != label_name]

    # Train decision tree
    tree = learn_decision_tree(train_examples, attributes, parent_examples=[])

    # Default label for unseen branches
    default_label = plurality_value(train_examples, label_name)

    # Evaluate function
    def evaluate(examples):
        correct = 0
        for ex in examples:
            pred = classify(tree, ex, default=default_label)
            if pred == ex[label_name]:
                correct += 1
        return correct / len(examples)

    train_acc = evaluate(train_examples)
    test_acc = evaluate(test_examples)

    # Print results
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test  Accuracy: {test_acc:.3f}")

    # Optional: show first 3 test predictions
    print("\nSample Predictions (first 3 test rows):")
    for ex in test_examples[:3]:
        pred = classify(tree, ex, default=default_label)
        clean_input = {k: ex[k] for k in ex if k != label_name}
        print(f" Input: {clean_input}")
        print(f" Pred:  {pred}   Actual: {ex[label_name]}\n")

    return tree, default_label, train_acc, test_acc


# ============================================================================
# Neural Network Model
# -------------------------
# Simple feedforward NN class
# -------------------------
class SimpleNN:
    """
    A simple Neural network with:
    - Input layer
    - Hidden layer with tanh activation
    - Output layer with sigmoid (0 - 1 probability)
    """

    # input_dim: number of input features
    # hidden_dim: number of neurons in hidden layer (more means more complex thinking)
    # lr: learning rate for gradient descent (slow learn = small lr, fast learn = large lr)
    # seed: random seed for reproducibility
    def __init__(self, input_dim, hidden_dim=8, lr=0.05, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        # Xavier-style initialization
        self.W1 = rng.normal(0, 1/np.sqrt(input_dim), (input_dim, hidden_dim)) # connects input to hidden
        self.b1 = np.zeros((1, hidden_dim)) # bias for hidden layer
        self.W2 = rng.normal(0, 1/np.sqrt(hidden_dim), (hidden_dim, 1)) # connects hidden to output
        self.b2 = np.zeros((1, 1)) # bias for output layer
        self.lr = lr # learning rate for gradient descent

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X):
        """
        Forward pass through the network. To make predictions.
        This is like asking the brain: "What do you think about these examples?"
        """
        # X: shape (n_samples, input_dim)
        # Input to hidden layer
        self.Z1 = X.dot(self.W1) + self.b1      # (n, hidden)
        self.A1 = np.tanh(self.Z1)              # activation
        # Hidden to output layer
        self.Z2 = self.A1.dot(self.W2) + self.b2  # (n,1)
        self.A2 = SimpleNN.sigmoid(self.Z2)       # predicted probabilities (n,1)
        return self.A2

    def backward(self, X, y):
        """
        Backward pass to learn from mistakes. Update weights using gradient descent.
        Network here will learn by adjusting weights.

        We wii: 
        - Compare predicted output (self.A2) to true labels (y)
        - Compute gradients of loss w.r.t weights
        - Figure out which weights to adjust to reduce error
        - Adjust weights slightly in direction that reduces error
        """
        # y shape (n,1)
        # Calculate gradients / errors at output layer
        n = X.shape[0] # number of examples
        # dLoss/dZ2 for BCE with sigmoid output = (A2 - y) / n
        dZ2 = (self.A2 - y) / n                 # (n,1)
        dW2 = self.A1.T.dot(dZ2)                # (hidden,1)
        db2 = np.sum(dZ2, axis=0, keepdims=True)   # (1,1)

        # Backpropagate to hidden layer
        dA1 = dZ2.dot(self.W2.T)                # (n, hidden)
        dZ1 = dA1 * (1 - np.tanh(self.Z1)**2)   # derivative of tanh

        dW1 = X.T.dot(dZ1)                      # (input_dim, hidden)
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1, hidden)

        # Updating weights and biases using gradient descent
        # move W2, b2, W1, b1 in direction that reduces loss
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, y, epochs=300, verbose=False):
        """
        Train the neural network - repeat forward and backward passes.
        X: input features (n_samples, input_dim) for training
        y: true labels (n_samples, 1)
        epochs: number of training iterations
        verbose: if True, print progress
        """
        # y expected shape (n,1)
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            if verbose and (epoch % 100 == 0 or epoch == epochs-1):
                loss = -np.mean(y * np.log(np.clip(self.A2, 1e-9, 1-1e-9)) +
                                (1 - y) * np.log(np.clip(1-self.A2, 1e-9, 1-1e-9)))
                print(f"Epoch {epoch:4d}  loss={loss:.4f}")

    def predict_proba(self, X):
        out = self.forward(X)
        return out.ravel()

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int).reshape(-1, 1)


def test_neural_network(train_df, test_df,
                        hidden_dim=12, lr=0.5, epochs=600, seed=42,
                        categorical_columns=None, label_name="WillWait",
                        show_n_samples=3, verbose=False):
    """
    Complete pipeline: Prepare data, train NN, evaluate, and show sample predictions.
    Args:
        train_df: DataFrame for training
        test_df: DataFrame for testing
        hidden_dim: number of neurons in hidden layer
        lr: learning rate
        epochs: number of training iteratinos
        seed: random seed for reproducibility
        categorical_columns: list of categorical columns to one-hot encode (auto-detected if None)
        label_name: name of the label column
        show_n_samples: number of test samples to display predictions for
        verbose: if True, print training progress
    """
    # Basic checks (Validate inputs)
    if label_name not in train_df.columns or label_name not in test_df.columns:
        raise ValueError(f"Both train_df and test_df must contain '{label_name}' column")

    # Auto-detect categorical columns if not provided
    if categorical_columns is None:
        categorical_columns = []
        for c in ['Type', 'WaitEstimate', 'Patrons', 'Price', 'Day']:
            if (c in train_df.columns) or (c in test_df.columns):
                categorical_columns.append(c)

    # One-hot encode categorical variables. So turn French/Italian/Thai/Burger into 4 binary columns such as [1,0,0,0]

    # Concatenate so get_dummies produces identical columns for train & test
    combined = pd.concat([train_df.reset_index(drop=True), test_df.reset_index(drop=True)], axis=0, ignore_index=True)

    # One-hot encode selected categorical columns (safe even if list empty)
    combined_enc = pd.get_dummies(combined, columns=categorical_columns, prefix=categorical_columns, drop_first=False)

    # Split encoded back to train/test by lengths
    n_train = len(train_df)
    train_enc = combined_enc.iloc[:n_train].reset_index(drop=True)
    test_enc  = combined_enc.iloc[n_train:].reset_index(drop=True)

    # Prepare features matrix and labels
    feature_names = [c for c in combined_enc.columns if c != label_name]
    X_train = train_enc[feature_names].values.astype(float)
    y_train = train_enc[label_name].values.reshape(-1, 1).astype(float)
    X_test  = test_enc[feature_names].values.astype(float)
    y_test  = test_enc[label_name].values.reshape(-1, 1).astype(float)

    # All prepared, now build and train the NN
    # BUILD & TRAIN NN
    # Build and train the NN (uses your SimpleNN class)
    model = SimpleNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim, lr=lr, seed=seed)
    model.fit(X_train, y_train, epochs=epochs, verbose=verbose)

    # Make predictions on train and test sets
    # Predict and evaluate
    yhat_train = model.predict(X_train)    # make predictions on train set, shape (n_train,1)
    yhat_test  = model.predict(X_test)     # make predictions on test set, shape (n_test,1)
    train_acc = float((yhat_train == y_train).mean())
    test_acc  = float((yhat_test  == y_test ).mean())

    # Print results in friendly format (like test_decision_tree)
    print(f"\nNeural Network results (hidden={hidden_dim}, lr={lr}, epochs={epochs}):")
    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test  Accuracy: {test_acc:.3f}")

    # Show sample test predictions
    print("\nSample Predictions (first {} test rows):".format(min(show_n_samples, len(y_test))))
    for i, (x_row, pred, actual) in enumerate(zip(X_test[:show_n_samples], yhat_test[:show_n_samples], y_test[:show_n_samples])):
        # optionally show the decoded original row by reading from test_df
        orig_row = test_df.reset_index(drop=True).iloc[i].to_dict()
        # remove label for display
        clean_input = {k: v for k, v in orig_row.items() if k != label_name}
        print(f"\n Input: {clean_input}")
        print(f" Pred:  {int(pred[0])}   Actual: {int(actual[0])}")

    # Return everything user might want
    return {
        'model': model,
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': feature_names,
        'train_acc': train_acc, 'test_acc': test_acc,
        'train_df_enc': train_enc, 'test_df_enc': test_enc
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    print("\n=== STEP 1: GENERATE 100 EXAMPLES (IN MEMORY) ===\n")
    df = generate_restaurant_data(n_samples=100, balance=True)

    # Show stats
    display_data_statistics(df)

    print("\n=== STEP 2: TRAIN/TEST SPLITS (IN MEMORY) ===\n")

    # Try ratios requested by assignment
    for ratio in [0.2, 0.3, 0.4]:
        print(f"\n--- Train Ratio {ratio*100:.0f}% ---")

        train_df, test_df = create_train_test_split(df, train_ratio=ratio)

        print(f"Train Size: {len(train_df)}")
        print(f"  - Yes: {sum(train_df['WillWait'])}")
        print(f"  - No:  {sum(~train_df['WillWait'])}")

        print(f"Test Size: {len(test_df)}")
        print(f"  - Yes: {sum(test_df['WillWait'])}")
        print(f"  - No:  {sum(~test_df['WillWait'])}")

        # You will later plug these into your:
        #   - Decision Tree training
        #   - Neural Network training
        # when you complete those parts.

        print(f"\n{'-'*70}")
        print("DECISION TREE MODEL")
        print(f"{'-'*70}")
        tree, default_label, train_acc, test_acc = test_decision_tree(train_df, test_df)

        print(f"\n{'-'*70}")
        print("NEURAL NETWORK MODEL")
        print(f"{'-'*70}")
        categorical_columns = ['Type', 'WaitEstimate', 'Patrons', 'Price']
        nn_res = test_neural_network(
            train_df=train_df,
            test_df=test_df,
            hidden_dim=12,
            lr=0.5,
            epochs=600,
            seed=42,
            categorical_columns=categorical_columns,
            label_name="WillWait",
            show_n_samples=3,
            verbose=False
        )

    print("\n=== DONE (All data stored in memory only) ===")

# Each item has 3 features: [sweetness, seeds, grows_underground]
# Values are 0 to 1. This is the training data.
training_data = [
    ('apple',      [0.8, 1.0, 0], True),
    ('banana',     [0.9, 0.2, 0], True),
    ('grape',      [0.8, 0.6, 0], True),
    ('strawberry', [0.8, 0.9, 0], True),
    ('avocado',    [0.2, 1.0, 0], True),
    ('tomato',     [0.3, 0.7, 0], True),
    ('carrot',     [0.3, 0.0, 1], False),
    ('broccoli',   [0.1, 0.0, 0], False),
    ('bread',      [0.2, 0.0, 0], False),
    ('cheese',     [0.1, 0.0, 0], False),
    ('potato',     [0.1, 0.0, 1], False),
]

def distance(a, b):
    return sum((a[i] - b[i]) ** 2 for i in range(len(a))) ** 0.5

def classify(item_features, k=3):
    distances = []
    for name, features, is_fruit in training_data:
        d = distance(item_features, features)
        distances.append((d, name, is_fruit))
    distances.sort()
    nearest = distances[:k]
    votes = sum(1 for d, name, is_fruit in nearest if is_fruit)
    prediction = votes > k / 2
    return prediction, nearest

# New items the AI has never seen before:
test_items = [
    ('orange', [0.7, 0.5, 0], True),
    ('corn',   [0.3, 0.4, 0], False),
]

for name, features, actual in test_items:
    prediction, nearest = classify(features)
    guess = 'Fruit' if prediction else 'Not Fruit'
    correct = 'Correct!' if prediction == actual else 'Wrong'
    print(name, '->', guess, '(' + correct + ')')
    print('   closest matches:', [n[1] for n in nearest])

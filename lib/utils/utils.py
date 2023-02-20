import pandas as pds

def oversample_data(df):
    largest_rep = max(df['label'].value_counts().tolist())
    other_entries = df[df['label'] != 'atis_flight']

    labels = other_entries['label'].unique().tolist()
    for label in labels:
        class_entries = df[df['label'] == label]
        num = len(class_entries)
        scale_factor = largest_rep // num
        print(num)
        print(scale_factor)
        df = df.append([class_entries]*scale_factor)
    return df

def ratio_undersample(df, frac):
    """
    Undersamples by a fraction rather than by class
    """
    n_samples = int(len(df)*frac)
    sub = df.sample(n=n_samples, random_state=1)
    return sub

def undersample_data(df):
    largest_set = df[df['label'] == 'atis_flight']
    other_entries = df[df['label'] != 'atis_flight']
    sub = largest_set.sample(n=500,random_state=1)
    print(sub)
    other_entries = other_entries.append([sub])
    return other_entries
import os
import numpy as np
import pandas as pd


def generate_user_vectors(
                  num_users=1000,
                  user_vector_dims=768,
                  target_values=["low", "high"],
                  split_ratios=[0.5, 0.5],
                  centers=[0, 5],
                  email="user{id}@mail.com"
                 ):
    """"
    Generate user vectors from normal distributions with different centers,
    and assign a target value to each group.
    """
    # Generate data and assign targets
    dfs = []
    for i, center in enumerate(centers):
        data_shape = (int(num_users * split_ratios[i]), user_vector_dims)
        generated_vectors = np.random.normal(center, size=data_shape)
        df = pd.DataFrame(generated_vectors)
        df["target"] = target_values[i]
        dfs.append(df)
    user_vectors = pd.concat(dfs)

    # Assign dummy emails (user1@mail.com, etc.)
    emails = [email.format(id=i) for i in range(num_users)]
    user_vectors["email"] = emails

    return user_vectors


if __name__ == "__main__":
    user_vectors = generate_user_vectors()

    # Split to party A and party B data
    party_a_data = user_vectors.drop("target", axis=1)
    party_b_data = user_vectors[["email", "target"]]

    # Generate pseudo internal email hashes
    party_a_hashes_df = pd.DataFrame(party_a_data["email"])
    party_a_hashes = ["party_a_hash{id}".format(id=i) for i in range(len(party_a_data))]
    party_a_hashes_df.insert(loc=0, column="hash", value=party_a_hashes)
    party_b_hashes_df = pd.DataFrame(party_b_data["email"])
    party_b_hashes = ["party_b_hash{id}".format(id=i) for i in range(len(party_b_data))]
    party_b_hashes_df.insert(loc=0, column="hash", value=party_b_hashes)

    # Replace user emails with pseudo hashes
    party_a_data.drop("email", axis=1, inplace=True)
    party_a_data.insert(loc=0, column="user_id", value=party_a_hashes)
    party_b_data.drop("email", axis=1, inplace=True)
    party_b_data.insert(loc=0, column="user_id", value=party_b_hashes)

    # Save data and email mappings to csv files
    save_dir = "examples/user_vectors/data"
    party_a_data.to_csv(os.path.join(save_dir, "party_a_user_vectors.csv"), index=False)
    party_b_data.to_csv(os.path.join(save_dir, "party_b_targets.csv"), index=False)
    party_a_hashes_df.to_csv(os.path.join(save_dir, "party_a_map.csv"), index=False)
    party_b_hashes_df.to_csv(os.path.join(save_dir, "party_b_map.csv"), index=False)

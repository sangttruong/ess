import numpy as np
import torch


class ESSampler:
    def __init__(
        self,
        model: Model,
        data,
        device,
    ):
        self.device = device
        self.model = model
        self.data = data

    def sample(
        self,
        sampling_index: int,
    ):
        """
        Write docstring

        Args:
            sampling_index (int): index where the sample gets updated
        """
        assert sampling_index < self.model.n_latent

        # WIP: Need to have previous state to move as Markov Chain
        nu = self.model.sample_prior(sampling_index)
        

        ll_current = self.likelihood.log_likelihood(
            previous_f[: self.train_test_split_idx],
            other_f[: self.train_test_split_idx],
            self.y_train,
        )
        ll_thres = ll_current + torch.log(torch.rand(1, device=self.device))

        angle = torch.rand(1, device=self.device) * 2 * np.pi
        angle_min, angle_max = angle - 2 * np.pi, angle

        while True:
            next_f = torch.cos(angle) * previous_f + torch.sin(angle) * nu
            log_likelihood = self.likelihood.log_likelihood(
                next_f[: self.train_test_split_idx],
                other_f[: self.train_test_split_idx],
                self.y_train,
            )

            if log_likelihood > ll_thres:
                break
            else:
                if angle == 0:
                    next_f = previous_f
                    break

                if angle < 0:
                    angle_min = angle
                else:
                    angle_max = angle
                angle = (
                    torch.rand(1, device=self.device) * (angle_max - angle_min)
                    + angle_min
                )

        # Save thetas and zs
        if sampling_theta:
            self.list_thetas.append(next_f.to(torch.bfloat16).cpu())
            self.previous_thetas = next_f

            self.previous_points = (
                torch.cos(angle) * self.previous_points + torch.sin(angle) * nu_points
            )
            self.list_points.append(self.previous_points.to(torch.bfloat16).cpu())
            self.iter += 1
            if self.iter % 100 == 0:
                print(f"Iteration {self.iter}\tLog likelihood: {log_likelihood}")

        elif sampling_z:
            constructed_z = torch.zeros((self.n_testcases,), device=self.device)
            constructed_z[self.all_squidx] = next_f
            self.list_zs.append(constructed_z.to(torch.bfloat16).cpu())
            self.previous_zs = next_f

        return log_likelihood
    
    def save_state(self, result_folder, iteration):
        # Save thetas and zs with torch.save
        torch.save(
            self.list_points,
            f"{result_folder}/ess_points_by_iter_{iteration}.pt",
        )
        torch.save(
            self.list_thetas,
            f"{result_folder}/ess_thetas_by_iter_{iteration}.pt",
        )
        torch.save(
            self.list_zs,
            f"{result_folder}/ess_zs_by_iter_{iteration}.pt",
        )
        print(f"Saved thetas and zs at iteration {iteration}")
        # Clear thetas and zs

        self.list_points = []
        self.list_thetas = []
        self.list_zs = []

    def load_state(self, result_folder, continue_iter=0):
        if continue_iter == 0:
            print("Initializing thetas and zs")
            self.iter = 0
            self.previous_thetas, self.previous_points = self.sample_theta_prior(
                sample_size=64
            )
            self.previous_zs = self.sample_z_prior()

        else:
            print("Loading thetas and zs")
            self.iter = continue_iter
            list_points = torch.load(
                f"{result_folder}/ess_points_by_iter_{continue_iter}.pt"
            )
            list_thetas = torch.load(
                f"{result_folder}/ess_thetas_by_iter_{continue_iter}.pt"
            )
            list_zs = torch.load(f"{result_folder}/ess_zs_by_iter_{continue_iter}.pt")

            self.previous_points = list_points[-1].to(self.device).float()
            self.previous_thetas = list_thetas[-1].to(self.device).float()
            self.previous_zs = list_zs[-1].to(self.device)[self.all_squidx].float()



from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sampling_prior(self):
        pass

    @abstractmethod
    def eval_log_likelihood(self):
        pass
    


class IRTModel(Model):
    def sample_z_prior(self):
        return self.z_prior_dists.sample()[self.all_squidx]

    def sample_theta_prior_by_student(self, sidx, sample_size=1):
        if self.theta_prior_dists[sidx] is None:
            return []
        else:
            return self.theta_prior_dists[sidx].sample(torch.Size([sample_size]))

    def sample_theta_prior(self, sample_size=1):
        thetas = []
        points = []
        for sidx in range(self.n_students):
            if self.list_saidx[sidx] is None:
                continue

            prior_samples = self.sample_theta_prior_by_student(sidx).mean(dim=0)

            thetas.append(prior_samples[: -self.n_points][self.list_saidx[sidx]])
            points.append(prior_samples[-self.n_points :])

        return torch.cat(thetas), torch.cat(points)

if __name__ == "__main__":
    #Placeholder values for n_students, list_saidx, student_idxs
    n_students = 100 #This is a placeholder
    list_saidx = [None] * n_students #This is a placeholder
    student_idxs = torch.rand(100) #This is a placeholder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_masks = []
    for sidx in range(n_students):
        if list_saidx[sidx] is None:
            student_masks.append(None)
        else:
            student_masks.append(student_idxs == sidx)

    # Initialize likelihood
    # likelihood = likelihood(device=device)
    list_points = []
    list_thetas = []
    list_zs = []

git fetch origin
git checkout 1-separate-model-from-sample
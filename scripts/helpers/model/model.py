
class NreCModel:
    def __init__(
        self,
        dir_,
        device,
        parameter_size: int,
        observable_size:int,
        num_hidden_layers:int,
        hidden_layer_size:int,
        activation,
    ):
        self.device = device
        self.architecture = MLP(
            parameter_size=parameter_size,
            observable_size=observable_size,
            num_hidden_layers=num_hidden_layers,
            hidden_layer_size=hidden_layer_size,
            activation=activation,
        )

        self.dir = Path(dir_)
        self.dir.mkdir(exist_ok=True)

    def load_weights(self, path: Path|str):
        weights_path = self.dir.joinpath("weights.pt")
        self.architecture.load_state_dict(load(weights_path), weights_only=True)

    def save_weights(self):
        weights_path = self.dir.joinpath("weights.pt")
        save(self.architecture.state_dict(), weights_path)
    
    def train(
        self,
        train_dataset: Dataset,
        optimizer,
        num_epochs,
        batch_size,
        k=10,
        gamma=1,
        val_dataset=None,
        lr_scheduler=None,
    ):
        train_dataloader = NreCDataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            k=k,
        )
        val_dataloader = (
            None if val_dataset is None 
            else NreCDataloader(
                dataset=val_dataset,
                batch_size=batch_size,
                k=k
            )
        )

        loss_fn = NreCLoss(gamma=gamma, k=k)

        self.architecture = self.architecture.to(device)
        losses = train_val(
            train_dataloader=train_dataloader,
            model=self.architecture,
            loss_fn=loss_fn,
            optimizer=optimizer,
            num_epochs=num_epochs,
            val_dataloader=val_dataloader,
            lr_scheduler=lr_scheduler,
        )

        self.save_weights()

        fig, ax = subplots()
        epochs = arange(len(losses))
        ax.scatter(losses["train"], epochs=epochs, label="Train.")
        ax.scatter(losses["val"], epochs=epochs, label="Val.")
        ax.legend()
        plot_path = self.dir.joinpath("loss.png")
        save_fig_and_close(fig, plot_path)

    def eval(
        self, 
        dataset, 
        scan_trial_nums, 
        linearity_trial_nums, 
        bounds,
        scan_size,
        rerun=True,
    ):
        self.run_likelihood_scan_2d(
            dataset,
            scan_trial_nums,
            bounds, 
            scan_size,
            rerun=rerun
        )
        self.run_linearity(
            dataset,
            linearity_trial_nums,
            bounds,
            scan_size,
            rerun=False
        )

    def run_likelihood_scan_2d(
        self,
        dataset,
        trial_nums,
        bounds,
        scan_size,
        rerun=True,
    ):
        if rerun:
            self._likelihood_scan_2d(
                dataset,
                trial_nums,
                bounds,
                scan_size,
            )

        result_path = self.dir.joinpath("scan_2d.npz") 
        result = load(result_path)
        likelihoods = result["likelihoods"]
        parameters = result["parameters"]
        mles = result["mles"]
        true_parameters = result["true_parameters"]

        parameters = parameters.reshape(scan_size, scan_size, -1)
        likelihoods = likelihoods.reshape(-1, scan_size, scan_size)

        for like, mle, truth, trial in zip(
            likelihoods, 
            mles, 
            true_parameters, 
            trial_nums
        ):
            fig, ax = subplots()
            ax.pcolormesh(parameters[:,:,0], parameters[:,:,1], like)
            ax.scatter(*truth, label="Truth")
            ax.scatter(*mle, label="MLE (Scan)")
            ax.legend()
            plot_path = self.dir.joinpath(f"scan_2d_{trial}.png")
            save_fig_and_close(fig, plot_path)

    def run_linearity(
        self,
        dataset,
        trial_nums,
        bounds,
        scan_size,
        rerun=True,
    ):
        if rerun:
            self._likelihood_scan_2d(
                dataset,
                trial_nums,
                bounds,
                scan_size,
            )

        result_path = self.dir.joinpath("scan_2d.npz")
        result = load(result_path)
        mles = result["mles"]
        true_parameters = result["true_parameters"]

        num_parameters = true_parameters.shape[1]
        for param_index, name in zip (range(num_parameters), ("7", "9"):
            true_param = true_values[:][param_index]
            mle = mles[:][param_index]
            bounds_ = bounds[param_index]
            fig, ax = subplots()
            ax.scatter(true_param, mle, color="#ff7f00", zorder=-50)
            ax.plot(bounds_, bounds_)
            ax.set_ylabel(r"Predicted $\delta C_" f"{name}" "$", fontsize=16)
            ax.set_xlabel(r"True $\delta C_" f"{name}" "$", fontsize=16)
            ax.legend(loc="upper left")
            plot_path = self.dir.joinpath(f"lin_c{name}.png")
            save_fig_and_close(fig, plot_path)
            
    def _likelihood_scan_2d(
        self,
        dataset,
        trial_nums,
        bounds,
        scan_size, 
    ):
        self.architecture = self.architecture.to(device) 
        self.architecture.eval()
        
        true_parameters = [dataset.trial_parameters(t) for t in trial_nums]
        true_parameters = [
            p.drop_duplicates().squeeze().numpy(force=True) 
            for p in true_parameters
        ]
        true_parameters = array(true_parameters)

        observables = [dataset.trial_observables(t) for t in trial_nums]
        observables = [tensor_from_pandas(o, dtype="float32") for o in observables]

        parameters = cartesian_prod(
            linspace(*bounds[0], scan_size), 
            linspace(*bounds[1], scan_size)
        )
        parameters = scan.to(dtype=float32, device=self.device)
        
        likelihoods = scan_over(model=model, parameters=parameters, observables=observables)
        mles, _ = mle(likelihoods)

        likelihoods = likelihoods.numpy(force=True)
        parameters = parameters.numpy(force=True)
        mles = mles.numpy(force=True)

        result_path = self.dir.joinpath("scan_2d.npz")
        savez(
            result_path, 
            likelihoods=likelihoods, 
            parameters=parameters, 
            mles=mles,
            true_parameters=true_parameters,
        )

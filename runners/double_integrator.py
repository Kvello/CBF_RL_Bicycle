from .base import BaseRunner
from envs.integrator import MultiObjectiveDoubleIntegratorEnv
class DoubleIntegratorRunner(BaseRunner):
    def setup(self):
        #######################
        # Environment:
        #######################
        base_env = MultiObjectiveDoubleIntegratorEnv(batch_size=args.get("num_parallel_env"),
                                                    device=device,
                                                    td_params=parameters,
                                                    seed=args["seed"])
        obs_signals = ["x1","x2"]
        ref_signals = ["y1_ref","y2_ref"]
        transforms = [
                UnsqueezeTransform(in_keys=obs_signals+ref_signals, 
                                dim=-1,
                                in_keys_inv=obs_signals+ref_signals,),
                CatTensors(in_keys=obs_signals, out_key= "obs",del_keys=False,dim=-1),
                CatTensors(in_keys=ref_signals, out_key= "ref",del_keys=False,dim=-1),
                ObservationNorm(in_keys=["obs"], out_keys=["obs"]),
                ObservationNorm(in_keys=["ref"], out_keys=["ref"]),
                CatTensors(in_keys=["obs","ref"], out_key="obs_extended",del_keys=False,dim=-1),
                DoubleToFloat(),
                StepCounter(max_steps=args["max_rollout_len"])]
        env = TransformedEnv(
            base_env,
            Compose(
                *transforms
            )
        ).to(device)
        env.transform[3].init_stats(num_iter=1000,reduce_dim=(0,1),cat_dim=1)
        env.transform[4].init_stats(num_iter=1000,reduce_dim=(0,1),cat_dim=1)
        
        #######################
        # Models:
        #######################


        nn_net_config = {
            "name": "feedforward",
            "eps": 1e-2,
            "layers": [64, 64],
            "activation": nn.ReLU(),
            "device": device,
            "input_size": len(obs_signals),
            "bounded": True,
        }
        actor_net = nn.Sequential()
        layers = [len(ref_signals+obs_signals)] + nn_net_config["layers"] 
        for i in range(len(layers)-1):
            actor_net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i + 1],device=device))
            actor_net.add_module(f"activation_{i}", nn_net_config["activation"])
        actor_net.add_module("output", nn.Linear(layers[-1], 2*env.action_spec.shape[-1],device=device))
        actor_net.add_module("param_extractor", NormalParamExtractor())


        policy_module = ProbabilisticActor(
            module=TensorDictModule(actor_net,
                                    in_keys=["obs_extended"],
                                    out_keys=["loc", "scale"]),
            in_keys=["loc", "scale"],
            spec=env.action_spec,
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": -parameters["max_input"],
                "high": parameters["max_input"],
            },
            return_log_prob=True,
        )
        if args.get("load_policy") is not None:
            policy_module.load_state_dict(torch.load(args.get("load_policy")))
            print("Policy loaded")

        CDF_net = CDFFactory.create(**nn_net_config)
        CDF_module = ValueOperator(
            module=CDF_net,
            in_keys=["obs"],
            out_keys=["V1"],
        )
        if args.get("load_CBF") is not None:
            CDF_module.load_state_dict(torch.load(args.get("load_CBF")))
            print("CBF network loaded") 

        value_net = nn.Sequential()
        layers = [len(ref_signals+obs_signals)] + nn_net_config["layers"]
        for i in range(len(layers)-1):
            value_net.add_module(f"layer_{i}", nn.Linear(layers[i], layers[i + 1],device=device))
            value_net.add_module(f"activation_{i}", nn_net_config["activation"])
        value_net.add_module("output", nn.Linear(layers[-1], 1,device=device))
        value_module = ValueOperator(
            module=value_net,
            in_keys=["obs_extended"],
            out_keys=["V2"]
        )

        

    def train(self):
        # Training code specific to the double integrator
        pass

    def evaluate(self):
        # Evaluation code specific to the double integrator
        pass

    def save(self):
        # Save code specific to the double integrator
        pass

    def load(self, 
             CDF_path:Optional[str]=None, 
             policy_path:Optional[str]=None, 
             value_path:Optional[str]=None):
         if value_path is not None:
            self.value_module.load_state_dict(torch.load(value_path))
            print("Value network loaded")       
        if policy_path is not None:
            self.policy_module.load_state_dict(torch.load(policy_path))
            print("Policy loaded")
        if CDF_path is not None:
            self.CDF_module.load_state_dict(torch.load(CDF_path))
            print("CBF network loaded")
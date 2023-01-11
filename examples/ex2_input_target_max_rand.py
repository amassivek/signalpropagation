from examples import *

def main():

    checkpoint = SaveModel.load()

    dataset_info, dataset_train, dataset_test = build_dataset()
    train_loader, test_loader = build_dataloader(
        dataset_train, dataset_test
    )
    print("dataset_info", dataset_info)

    info = argparse.Namespace()

    if args.sigprop:
        input_ch = None

        sp_manager = sigprop.managers.Preset(
            sigprop.models.Forward,
            sigprop.propagators.Loss,
            sigprop.propagators.signals.Loss,
            build_optimizer
        )
        input_shape = (dataset_info.num_classes,)
        input_ch = 128
        output_shape = (input_ch, dataset_info.input_dim, dataset_info.input_dim)
        sp_signal = sigprop.signals.ProjectionContextInput(
            *signal_modules_lpi(input_shape, output_shape),
            input_shape, output_shape
        )
        sp_signal = nn.Sequential(
            sigprop.signals.LabelNumberToOnehot(
                dataset_info.num_classes
            ),
            sp_signal
        )
        sp_manager.config_propagator(
            loss=sigprop.loss.v14_input_target_max_rand
        )
        sp_manager.set_signal(
            sp_signal,
            loss=sigprop.loss.v14_input_target_max_rand
        )

        runner = RunnerSigprop(monitor_main)

    else:
        sp_manager = None
        input_ch = None

        runner = Runner()

    model = build_model(dataset_info, sp_manager, input_ch)

    model = ModelWrapper(model, dataset_info)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])

    pprint(vars(args))
    print(model)
    print('[Global Loss] Model {} has {} parameters'.format(args.model, count_parameters(model)))
    print("cuda", args.cuda)

    runner.train(
        model,
        dataset_info.num_classes,
        train_loader, test_loader,
        info
    )

main()


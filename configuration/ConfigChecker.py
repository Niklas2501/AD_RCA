from configuration.Configuration import Configuration


class ConfigChecker:

    def __init__(self, config: Configuration, dataset, architecture, training):
        self.config: Configuration = config
        self.dataset = dataset
        self.architecture_type = architecture
        self.training = training
        self.list_of_warnings = []

    @staticmethod
    def implication(p, q, error):
        # p --> q == !p or q
        assert not p or q, error

    # Can be used to define forbidden / impossible parameter configurations
    # and to output corresponding error messages if they are set in this way.
    def pre_init_checks(self):
        assert self.architecture_type in ['anomalyDetection',
                                          'preprocessing'], 'invalid architecture passed to configChecker'

        if self.architecture_type == 'preprocessing':
            self.warnings()

    @staticmethod
    def print_warnings(warnings):
        print()
        print('##########################################')
        print('WARNINGS:')
        for warning in warnings:
            if type(warning) == str:
                print('-  ' + warning)
            elif type(warning) == list:
                print('-  ' + warning.pop(0))
                for string in warning:
                    print('   ' + string)
        print('##########################################')
        print()

    # Add entries for which the configuration is valid but may lead to errors or unexpected behaviour
    def warnings(self):

        if not self.config.use_hyper_file:
            self.list_of_warnings.append(['Hyperparameters shouldn\'t be read from file. ',
                                          'Ensure entries in Hyperparameters.py are correct.'])

        if len(self.list_of_warnings) > 0:
            self.print_warnings(self.list_of_warnings)

    def post_init_checks(self, architecture):

        self.warnings()

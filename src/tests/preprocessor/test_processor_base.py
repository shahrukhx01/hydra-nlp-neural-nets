'''
Text Preprocessor Unit Test Base
Author: Shahrukh Khan
'''
import unittest

class TestPreprocessorBase(unittest.TestCase):
    """ Base class for Test Preprocessor to supply parameters to Unit Test Class
    """
    def __init__(self, methodName='test_preprocessor', preprocess_config=None, data=None, groundtruth=None, log=None):
        super(TestPreprocessorBase, self).__init__(methodName)

        self.preprocess_config = preprocess_config
        self.data = data
        self.groundtruth = groundtruth
        self.log = log


    @staticmethod
    def parametrize(test_processor, preprocess_config=None, data=None, groundtruth=None, log=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameters
        """
        # Get child names of this class
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(test_processor)
        suite = unittest.TestSuite()

        # Delegate passed parameter forward to child class
        for name in testnames:
            suite.addTest(test_processor(
                    name, preprocess_config=preprocess_config, data=data, groundtruth=groundtruth, log=log))

        return suite
 
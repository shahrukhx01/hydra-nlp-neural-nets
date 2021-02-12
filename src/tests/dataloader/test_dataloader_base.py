'''
Data Loader Unit Test Base
Author: Shahrukh Khan
'''
import unittest

class TestDataLoaderBase(unittest.TestCase):
    """ Base class for Test Data Loader to supply parameters to Unit Test Class
    """
    def __init__(self, methodName='test_preprocessor', config=None, groundtruth=None, log=None):
        super(TestDataLoaderBase, self).__init__(methodName)

        self.config = config
        self.groundtruth = groundtruth
        self.log = log


    @staticmethod
    def parametrize(test_data_loader, config=None, groundtruth=None, log=None):
        """ Create a suite containing all tests taken from the given
            subclass, passing them the parameters
        """
        # Get child names of this class
        testloader = unittest.TestLoader()
        testnames = testloader.getTestCaseNames(test_data_loader)
        suite = unittest.TestSuite()

        # Delegate passed parameter forward to child class
        for name in testnames:
            suite.addTest(test_data_loader(
                    name, config=config, groundtruth=groundtruth, log=log))

        return suite
 
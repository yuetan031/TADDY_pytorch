import abc

class setting:
    '''
    SettingModule: Abstract Class
    Entries: 
    '''
    
    setting_name = None
    setting_description = None
    
    dataset = None
    method = None
    result = None
    evaluate = None

    def __init__(self, sName=None, sDescription=None):
        self.setting_name = sName
        self.setting_description = sDescription
    
    def prepare(self, sDataset, sMethod):
        self.dataset = sDataset
        self.method = sMethod

    @abc.abstractmethod
    def run(self):
        return

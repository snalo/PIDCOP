class MZI:
    
    def __init__(self):
        self.no_of_parallel_requests = 1  # MZI typically handles one request at a time
        self.power= 2.25e-3  # (taken from #Lightening Transformer_https://ieeexplore.ieee.org/document/10476418/ )
        #self.power_eo = 1e-6  # electro-optic tuning power in watts (typical value for high-speed modulation)
        self.energy = 1e-12  # energy per operation in joules (1 pJ/bit for electro-optic MZI)
        self.latency = 1e-9  # latency in seconds (1 ns, typical for high-speed MZI modulators)
        self.area = 5.38e-3  # area in mm² (taken from #Lightening Transformer_https://ieeexplore.ieee.org/document/10476418/ )
        self.request_queue = None  # queue for incoming requests
        self.waiting_queue = None  # queue for waiting requests
        self.start_time = 0  # start time of the current operation
        self.end_time = 0  # end time of the current operation
        self.calls_count = 0  # counter for the number of operations performed


        
class WindType:
    def __init__(self, path):
        self._is_hrrr, self._is_rtma, self._type = self.type_from_path(path)
        self._alts = self.compute_wind_alts()

    def type_from_path(self, path):
        if "hrrr" in path.lower():
            is_hrrr = True
            is_rtma = False
            wind_type = "HRRR"
        elif "rtma" in path.lower():
            is_hrrr = False
            is_rtma = True
            wind_type = "RTMA"
        else:
            is_hrrr = False
            is_rtma = False
            wind_type = "UNKNOWN"
        return is_hrrr, is_rtma, wind_type
        
    def compute_wind_alts(self):
        if self.is_hrrr():
            alts = [10, 80]
        elif self.is_rtma():
            alts = [10]
        else:
            alts = []
        return alts
    
    def is_hrrr(self):
        return self._is_hrrr

    def is_rtma(self):
        return self._is_rtma

    def is_unknown(self):
        return not (self._is_hrrr or self._is_rtma)
    
    def type_as_str(self):
        return self._type

    def alts(self):
        return self._alts
    

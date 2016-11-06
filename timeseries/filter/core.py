
# Abstract class Filter
class Filter:
    def __init__(this):
        pass
        
    def Estep(this):
        this.expectation_delegate(this)

    def Mstep(this):
        this.maximization_delegate(this)

    def expectation_delegate(this):
        assert 0 "Not implemented expectation_delegate method in child class"

    def maximization_delegate(this):
        assert 0 "Not implemented maximization_delegate method in child class"

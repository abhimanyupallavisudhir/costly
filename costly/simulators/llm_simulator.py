from costly.simulators.faker_factory import FakerFactory
from costly.costlog import Costlog
from costly.estimators.llm_api_estimation import LLM_API_Estimation

class LLM_Simulator(FakerFactory):
    """
    See the docstring for FakerFactory.
    """
    
    @staticmethod
    def simulate_llm_call(
        input_string: str,
        model: str = None,
        response_model: type = str,
        cost_log: Costlog = None,
        description: list[str] = None,
    ) -> dict:
        """
        Simulate an LLM call.
        """
        response_model = response_model or str
        response = FakerFactory.fake(response_model)
        if cost_log is not None:
            assert model is not None, "model is required for tracking costs"
            with cost_log.new_item() as (item, _):
                cost_item = LLM_API_Estimation.get_cost_item_simulating(
                    input_string=input_string,
                    model=model,
                    description=description,
                    # output_string=response, # not needed
                )
                item.update(cost_item)
        return response

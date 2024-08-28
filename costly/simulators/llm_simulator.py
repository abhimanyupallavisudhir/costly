from faker import Faker
from costly.costlog import Costlog
from costly.estimators.llm_api_estimation import LLM_API_Estimation
from pydantic import BaseModel
from datetime import datetime


class LLM_Simulator:

    FAKER = Faker()

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
        response = LLM_Simulator.fake(response_model)
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

    @staticmethod
    def fake(t: type):
        if t == str:
            return LLM_Simulator.FAKER.text(
                max_nb_chars=int(600 * 4.5), ext_word_list=["delve"]
            )
        elif t == int:
            return LLM_Simulator.FAKER.random_int(min=0, max=100)
        elif t == float:
            return LLM_Simulator.FAKER.random_number(digits=3)
        elif t == bool:
            return LLM_Simulator.FAKER.random_element(elements=[True, False])
        elif t.__origin__ == list:
            return [
                LLM_Simulator.fake(t.__args__[0])
                for _ in range(LLM_Simulator.FAKER.random_int(min=0, max=10))
            ]
        elif t.__origin__ == dict:
            return {
                LLM_Simulator.fake(t.__args__[0]): LLM_Simulator.fake(t.__args__[1])
            }
        elif t.__origin__ == tuple:
            return tuple(
                LLM_Simulator.fake(t.__args__[i]) for i in range(len(t.__args__))
            )
        elif t == datetime:
            return LLM_Simulator.FAKER.date_time_this_decade()
        elif t == datetime.date:
            return LLM_Simulator.FAKER.date_this_decade()
        elif t == datetime.time:
            return LLM_Simulator.FAKER.time_object()
        elif issubclass(t, BaseModel):
            factory_dict = {}
            for name, field in t.model_fields.items():
                factory_dict[name] = LLM_Simulator.fake(field.type_)
            return t(**factory_dict)
        else:
            raise ValueError(f"Unsupported type: {t}")

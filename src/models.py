from pydantic import BaseModel
from datetime import datetime
from typing import Callable, Tuple, Optional
from  src.utils.get_profit import get_profit_percent




class SaleRow(BaseModel):
    name: str
    buy_price: float
    price_1: float
    price_2: float
    price_3: float
    price_4: float
    price_5: float
    price_6: float
    price_7: float
    price_8: float
    price_9: float
    price_10: float
    price_11: Optional[float] = None
    price_12: Optional[float] = None
    price_13: Optional[float] = None
    price_14: Optional[float] = None
    price_15: Optional[float] = None
    ts_1: int
    ts_2: int
    ts_3: int
    ts_4: int
    ts_5: int
    ts_6: int
    ts_7: int
    ts_8: int
    ts_9: int
    ts_10: int
    ts_11: Optional[int] = None   
    ts_12: Optional[int] = None
    ts_13: Optional[int] = None
    ts_14: Optional[int] = None
    ts_15: Optional[int] = None
    sold_price: Optional[float] = None
    date_added: Optional[datetime] = None

    @property
    def train_prices(self) -> list[float]:
        return [
            self.price_1, self.price_2, self.price_3, self.price_4, self.price_5,
            self.price_6, self.price_7, self.price_8, self.price_9, self.price_10
        ]

    @property
    def future_validated_prices(self) -> list[float]:
        return [self.price_11, self.price_12, self.price_13, self.price_14, self.price_15]
        


DecisionFunc = Callable[[SaleRow, dict], Tuple[bool, float]]


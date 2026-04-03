from typing import Optional, Literal
from pydantic import BaseModel, Field, conset

# Schema definition for JSON output
TRANSPORTATION_TYPES = Literal["Walking", "Public transportation", "Private vehicle"]

class CloseEntity(BaseModel):
    entity_name: str = Field(description="The name of entity", max_length=50)
    distance_in_km: Optional[int] = Field(description="The maximum distance from the entity to the property in km")
    distance_in_minute: Optional[int] = Field(description="The maximum time to travel from the entity to the property")
    transportation_type: Optional[TRANSPORTATION_TYPES] = Field(description="The type of transportation to travel to the entity")


PROPERTY_TYPES = Literal["Apartment", "Townhouse", "House", "Studio", "Unit"]

class PropertyDetails(BaseModel):
    area: str = Field(description="The area to search", max_length=50, default=None)
    min_rental_fee_per_week: int = Field(description="The minimum weekly rental fee", ge=0)
    max_rental_fee_per_week: int = Field(description="The maximum weekly rental fee", default=None)
    min_num_bedrooms: int = Field(description="The minimum number of bedrooms", ge=1)
    max_num_bedrooms: int = Field(description="The maximum number of bedrooms", le=10)
    min_num_bathrooms: int = Field(description="The minimum number of bathrooms", ge=1)
    min_num_carspaces: int = Field(description="The minimum number of carspaces", ge=0)
    property_type: conset(PROPERTY_TYPES, min_length=0, max_length=5)
    use_public_transporation: bool = Field(description="Whether the user mentions public transportation. Some specific public transportations are 'bus', 'train', 'tram', or combinations of them", default=False)
    close_to: list[CloseEntity] = Field(default=[])
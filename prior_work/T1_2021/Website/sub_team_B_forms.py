from flask_wtf import FlaskForm
from wtforms import StringField, RadioField, SelectField, PasswordField, SubmitField, BooleanField, IntegerField, FloatField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from wtforms.fields.html5 import DateField

class sub_team_B_form(FlaskForm):
	
	submit_Victoria_Point = SubmitField("Victoria Point Street")
	submit_Bourke_Street_Mall = SubmitField("Bourke Street Mall South Street")
	submit_Collins_Place = SubmitField("Collins Place Street")
	submit_Southern_Cross_Station = SubmitField("Southern Cross Station Street")
    submit_Flinders_St_Spark_La = SubmitField("Flinders St Spark La Street")
from flask_wtf import FlaskForm
from wtforms import StringField, RadioField, SelectField, PasswordField, SubmitField, BooleanField, IntegerField, FloatField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from wtforms.fields.html5 import DateField

class Pedestrian_prediction_Form(FlaskForm):
  
    date = DateField('Date', format='%d-%m-%Y', validators= [DataRequired()])
    rainfall = FloatField("Rainfall amount", validators=[DataRequired()])
    solar_exposure = FloatField("Solar exposure", validators=[DataRequired()])
    minimum_temperature = FloatField("Minimum temperature (Degree C)",  validators=[DataRequired()])
    maximum_temperature = FloatField("Maximum temperature (Degree C)",  validators=[DataRequired()])
    public_holiday = SelectField("Public holiday", choices = [(1,"Yes"), (0,"No")],validators=[DataRequired()])
    restriction = SelectField("Restriction", choices = [(1,"Yes"), (0,"No")],validators=[DataRequired()])
    submit = SubmitField("Predict")
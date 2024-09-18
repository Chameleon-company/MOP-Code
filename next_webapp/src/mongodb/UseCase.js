// models/UseCase.js
const mongoose = require('mongoose');

const UseCaseSchema = new mongoose.Schema({
    tag: {
        type: String,
        required: [true, 'Please provide a tag for this use case.'],
        maxlength: [30, 'Tag cannot be more than 30 characters'],
    },
    popularity: {
        type: Number,
        required: true,
        min: [0, 'Popularity must be a non-negative number'],
        max: [100, 'Popularity cannot be more than 100'],
    },
    semester: {
        type: String,
        required: [true, 'Please provide a semester for this use case.'],
        match: [/^\d{4}-T[1-3]$/, 'Semester must be in the format YYYY-T1, YYYY-T2, or YYYY-T3']
    },
});

module.exports = mongoose.models.UseCase || mongoose.model('UseCase', UseCaseSchema);
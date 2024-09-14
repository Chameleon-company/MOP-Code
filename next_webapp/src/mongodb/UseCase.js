const mongoose = require('mongoose');

const UseCaseSchema = new mongoose.Schema({
    tag: {
        type: String,
        required: [true, 'Please provide a tag for this use case.'],
        maxlength: [30, 'Tag cannot be more than 30 characters'],
    },
    publishNumber: {
        type: String,
        required: [true, 'Please provide a publish number for this use case.'],
    },
    popularity: {
        type: String,
        required: [true, 'Please provide a popularity percentage for this use case.'],
        match: [/^\d{1,3}%$/, 'Popularity must be a percentage in the format XX%']
    },
    trimester: {
        type: String,
        required: [true, 'Please provide a trimester for this use case.'],
        match: [/^[1-3]$/, 'Trimester must be one of "1", "2", or "3"']
    },
});

module.exports = mongoose.models.UseCase || mongoose.model('UseCase', UseCaseSchema);
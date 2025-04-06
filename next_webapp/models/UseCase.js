import mongoose from 'mongoose';

const UseCaseSchema = new mongoose.Schema({
    name: {
        type: String,
        required: true,
    },
    description: {
        type: String,
        required: true,
    },
    tags: {
        type: [String],
        required: true,
    },
    filename: {
        type: String,
        required: false
    }
});

export default mongoose.models.UseCase || mongoose.model('UseCase', UseCaseSchema);

      // export type CaseStudy = {
      //   id: number;
      //   name: string;
      //   description: string;
      //   tags: string[];
      //   filename?: string;
      // };
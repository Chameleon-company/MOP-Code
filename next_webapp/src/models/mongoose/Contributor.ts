import mongoose, { Schema, type InferSchemaType, type Model } from "mongoose";

import { TEAM_ROLES, TEAMS, ROLES, LEVELS, CONTRIBUTOR_TYPES, type TeamName } from "@/types/contributor";

const contributorSchema = new Schema(
  {
    name: { type: String, required: true, trim: true },
    year: { type: Number, required: true },
    trimester: { type: Number, required: true, enum: [1, 2, 3] },
    contributor_type: {
      type: String,
      required: true,
      enum: CONTRIBUTOR_TYPES,
    },

    // Only meaningful for students forced to null for mentors/directors
    // by the pre-validate hook below, regardless of what's submitted.
    team: { type: String, enum: [...TEAMS, null], default: null },
    position: { type: String, enum: [...ROLES, null], default: null },
    level: { type: String, enum: [...LEVELS, null], default: null },

    display_order: { type: Number, default: 0 },
    is_active: { type: Boolean, default: true },
  },
  {
    collection: "contributors",
    timestamps: { createdAt: "created_at", updatedAt: "updated_at" },
  },
);

// Student/mentor/director rules: only students carry team/position/level.
// Non-students are silently normalized to null (matches what the admin form
// already does client-side for mentor/director submissions); students must
// supply all three.
contributorSchema.pre("validate", function (next) {
  if (this.contributor_type !== "student") {
    this.team = null;
    this.position = null;
    this.level = null;
  } else {
    if (!this.team) {
      this.invalidate("team", "team is required for student contributors");
    }
    if (!this.position) {
      this.invalidate("position", "position is required for student contributors");
    }
    if (!this.level) {
      this.invalidate("level", "level is required for student contributors");
    }
    
    if (this.team && this.position) {
      const validRoles = TEAM_ROLES[this.team as TeamName];
      if (validRoles && !(validRoles as readonly string[]).includes(this.position)) {
        this.invalidate("position", `Position '${this.position}' is not valid for team '${this.team}'`);
      }
    }
  }
  next();
});

contributorSchema.index({ year: -1, trimester: 1, display_order: 1 });
contributorSchema.index({ contributor_type: 1 });
contributorSchema.index({ team: 1 });

export type ContributorDocument = InferSchemaType<typeof contributorSchema>;

export const Contributor =
  (mongoose.models.Contributor as Model<ContributorDocument>) ||
  mongoose.model<ContributorDocument>("Contributor", contributorSchema);

export default Contributor;

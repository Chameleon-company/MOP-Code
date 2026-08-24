"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

type CustomSelectOption = {
  value: string;
  label: string;
  icon?: string;
};

type CustomSelectProps = {
  value: string;
  onChange: (value: string) => void;
  options: CustomSelectOption[];
  placeholder?: string;
  disabled?: boolean;
  className?: string;
};

function CustomSelect({
  value,
  onChange,
  options,
  placeholder = "Select option",
  disabled = false,
  className = "",
}: CustomSelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () =>
      document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectedOption = options.find((opt) => opt.value === value);

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        disabled={disabled}
        className={`mt-2 w-full rounded-xl border border-gray-200 px-4 py-3 text-[15px] outline-none transition flex items-center justify-between ${
          disabled
            ? "cursor-not-allowed bg-gray-50 text-gray-400 opacity-60"
            : "cursor-pointer bg-white text-gray-800 hover:border-gray-300 focus:border-[#2DBE6C] focus:ring-2 focus:ring-[#2DBE6C]/10"
        } ${className}`}
      >
        <span
          className={
            selectedOption && selectedOption.value
              ? "text-gray-800"
              : "text-gray-400"
          }
        >
          {selectedOption && selectedOption.value
            ? selectedOption.label
            : placeholder}
        </span>
        <ChevronDown
          size={18}
          className={`text-gray-400 transition ${
            isOpen ? "rotate-180" : ""
          }`}
        />
      </button>

      {isOpen && !disabled && (
        <div className="absolute top-full left-0 right-0 mt-2 rounded-xl border border-gray-200 bg-white shadow-lg z-10">
          {options.map((option) => (
            <button
              key={option.value}
              type="button"
              onClick={() => {
                onChange(option.value);
                setIsOpen(false);
              }}
              className={`w-full px-4 py-3 text-left text-[15px] transition first:rounded-t-xl last:rounded-b-xl ${value === option.value
                  ? "bg-[#2DBE6C]/10 text-[#2DBE6C] font-medium"
                  : "text-gray-800 hover:bg-gray-50"
                }`}
            >
              <span>
                {option.icon && <span className="mr-2">{option.icon}</span>}
                {option.label}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export type ContributorFormData = {
  name: string;
  year: string;
  trimester: string;
  contributorType: "student" | "mentor" | "company_director";
  team: string;
  position: string;
  level: "Senior" | "Junior" | "";
  displayOrder: string;
  isActive: boolean;
};

type ContributorFormProps = {
  initialData?: Partial<ContributorFormData>;
  onSubmit: (data: ContributorFormData) => void | Promise<void>;
  submitting?: boolean;
  submitLabel?: string;
};

const currentYear = new Date().getFullYear();

const defaultFormData: ContributorFormData = {
  name: "",
  year: String(currentYear),
  trimester: "1",
  contributorType: "student",
  team: "",
  position: "",
  level: "",
  displayOrder: "0",
  isActive: true,
};

const inputClassName =
  "mt-2 w-full rounded-xl border border-gray-200 bg-white px-4 py-3 text-[15px] text-gray-800 outline-none transition focus:border-[#2DBE6C] focus:ring-2 focus:ring-[#2DBE6C]/10";

const labelClassName = "text-[14px] font-medium text-[#344054]";

export const TEAM_OPTIONS: CustomSelectOption[] = [
  { value: "", label: "Select team", icon: "" },
  { value: "Data Science Team", label: "Data Science Team", icon: "📊" },
  { value: "Website Development Team", label: "Website Development Team", icon: "💻" },
  { value: "Design Team", label: "Design Team", icon: "🎨" },
  { value: "Cyber Security Team", label: "Cyber Security Team", icon: "🛡️" },
];

export const TEAM_ROLES: Record<string, string[]> = {
  "Data Science Team": [
    "Data Scientist",
    "Data Science Team Lead",
    "Data Science Quality Manager",
  ],
  "Website Development Team": [
    "Web Developer",
    "Web Dev Team Lead",
    "Web Dev Quality Manager",
  ],
  "Design Team": [
    "Design Team Member",
    "Design Team Lead",
  ],
  "Cyber Security Team": [
    "Cyber Security Team Member",
    "Cyber Security Team Lead",
  ],
};

export default function ContributorForm({
  initialData,
  onSubmit,
  submitting = false,
  submitLabel = "Add Contributor",
}: ContributorFormProps) {
  const [formData, setFormData] =
    useState<ContributorFormData>(defaultFormData);

  const [validationError, setValidationError] = useState("");

  useEffect(() => {
    if (initialData) {
      setFormData({
        ...defaultFormData,
        ...initialData,
      });
    }
  }, [initialData]);

  const updateField = <K extends keyof ContributorFormData>(
    field: K,
    value: ContributorFormData[K],
  ) => {
    setFormData((previous) => ({
      ...previous,
      [field]: value,
    }));
  };

  const handleContributorTypeChange = (
    type: ContributorFormData["contributorType"],
  ) => {
    setFormData((previous) => ({
      ...previous,
      contributorType: type,
      team: "",
      position: "",
      level: "",
    }));
  };

  const handleTeamChange = (team: string) => {
    setFormData((previous) => ({
      ...previous,
      team,
      position: "",
    }));
  };

  const handleSubmit = async (
    event: FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();
    setValidationError("");

    if (!formData.name.trim()) {
      setValidationError("Contributor name is required.");
      return;
    }

    if (!formData.year) {
      setValidationError("Year is required.");
      return;
    }

    if (!formData.trimester) {
      setValidationError("Trimester is required.");
      return;
    }

    if (formData.contributorType === "student") {
      if (!formData.team.trim()) {
        setValidationError("Team is required for students.");
        return;
      }
      if (!formData.position.trim()) {
        setValidationError("Position or Role is required for students.");
        return;
      }
      if (!formData.level) {
        setValidationError("Level is required for students.");
        return;
      }
    }

    await onSubmit(formData);
  };

  const isStudent = formData.contributorType === "student";

  const availableRoles = formData.team ? TEAM_ROLES[formData.team] || [] : [];
  const roleOptions: CustomSelectOption[] = formData.team
    ? [
        { value: "", label: "Select position" },
        ...availableRoles.map((role) => ({ value: role, label: role })),
      ]
    : [{ value: "", label: "Select a team first" }];

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      {validationError && (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
          {validationError}
        </div>
      )}

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <div className="md:col-span-2">
          <label className={labelClassName}>
            Full Name <span className="text-red-500">*</span>
          </label>

          <input
            type="text"
            value={formData.name}
            onChange={(event) =>
              updateField("name", event.target.value)
            }
            placeholder="Enter full name"
            className={inputClassName}
            disabled={submitting}
          />
        </div>

        <div>
          <label className={labelClassName}>
            Year <span className="text-red-500">*</span>
          </label>

          <input
            type="number"
            min="2000"
            max="2100"
            value={formData.year}
            onChange={(event) =>
              updateField("year", event.target.value)
            }
            className={inputClassName}
            disabled={submitting}
          />
        </div>

        <div>
          <label className={labelClassName}>
            Trimester <span className="text-red-500">*</span>
          </label>

          <CustomSelect
            value={formData.trimester}
            onChange={(value) =>
              updateField("trimester", value)
            }
            options={[
              { value: "1", label: "Trimester 1" },
              { value: "2", label: "Trimester 2" },
              { value: "3", label: "Trimester 3" },
            ]}
            placeholder="Select trimester"
            disabled={submitting}
          />
        </div>

        <div>
          <label className={labelClassName}>
            Contributor Type{" "}
            <span className="text-red-500">*</span>
          </label>

          <CustomSelect
            value={formData.contributorType}
            onChange={(value) =>
              handleContributorTypeChange(
                value as ContributorFormData["contributorType"],
              )
            }
            options={[
              { value: "student", label: "👨‍🎓 Student" },
              { value: "mentor", label: "👨‍🏫 Mentor" },
              { value: "company_director", label: "🏢 Company Director" },
            ]}
            placeholder="Select type"
            disabled={submitting}
          />
        </div>

        {isStudent && (
          <div>
            <label className={labelClassName}>
              Team <span className="text-red-500">*</span>
            </label>

            <CustomSelect
              value={formData.team}
              onChange={handleTeamChange}
              options={TEAM_OPTIONS}
              placeholder="Select team"
              disabled={submitting}
            />
          </div>
        )}

        {isStudent && (
          <div>
            <label className={labelClassName}>
              Position or Role <span className="text-red-500">*</span>
            </label>

            <CustomSelect
              value={formData.position}
              onChange={(value) =>
                updateField("position", value)
              }
              options={roleOptions}
              placeholder={
                formData.team ? "Select position" : "Select a team first"
              }
              disabled={submitting || !formData.team}
            />
          </div>
        )}

        {isStudent && (
          <div>
            <label className={labelClassName}>
              Level <span className="text-red-500">*</span>
            </label>

            <CustomSelect
              value={formData.level}
              onChange={(value) =>
                updateField("level", value as "Senior" | "Junior" | "")
              }
              options={[
                { value: "", label: "Select level" },
                { value: "Senior", label: "Senior" },
                { value: "Junior", label: "Junior" },
              ]}
              placeholder="Select level"
              disabled={submitting}
            />
          </div>
        )}

        <div>
          <label className={labelClassName}>
            Display Order
          </label>

          <input
            type="number"
            min="0"
            value={formData.displayOrder}
            onChange={(event) =>
              updateField("displayOrder", event.target.value)
            }
            className={inputClassName}
            disabled={submitting}
          />

          <p className="mt-2 text-xs text-[#687280]">
            Lower numbers will appear first.
          </p>
        </div>

        <div>
          <label className={labelClassName}>
            Status
          </label>

          <label className="mt-2 flex min-h-[48px] cursor-pointer items-center gap-3 rounded-xl border border-gray-200 px-4">
            <input
              type="checkbox"
              checked={formData.isActive}
              onChange={(event) =>
                updateField("isActive", event.target.checked)
              }
              className="h-4 w-4 accent-[#2DBE6C]"
              disabled={submitting}
            />

            <span className="text-sm text-[#344054]">
              Show this contributor on the website
            </span>
          </label>
        </div>
      </div>

      <div className="flex justify-end border-t border-gray-100 pt-6">
        <button
          type="submit"
          disabled={submitting}
          className="rounded-xl bg-[#2DBE6C] px-7 py-3 text-sm font-semibold text-white transition hover:bg-[#25A85E] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {submitting ? "Saving..." : submitLabel}
        </button>
      </div>
    </form>
  );
}